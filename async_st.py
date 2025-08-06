from typing import Callable, Dict, Optional, List, Tuple
from flwr.server.strategy.strategy import Strategy
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateRes,
    EvaluateIns,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np

# Globals (set in your main script)
CURRENT_ROUND = 0
EXPERIMENT_LOG_PATH = "metrics.log"


class FedAsync(Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        learning_rate: float = 0.1,
        decay: float = 0.99,
        evaluate_fn: Optional[
            Callable[[int, Parameters], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
    ):
        self.parameters = initial_parameters
        self.learning_rate = learning_rate
        self.decay = decay
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.round = 0

    def __repr__(self) -> str:
        return "FedAsync"

    def initialize_parameters(self, client_manager):
        return self.parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(num_clients=1, min_num_clients=1)
        fit_ins = FitIns(self.parameters, {})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        global CURRENT_ROUND
        CURRENT_ROUND = server_round

        if not results:
            return self.parameters, {}

        client_res = results[0][1]  # Only one client in async mode

        old_weights = parameters_to_ndarrays(self.parameters)
        new_weights = parameters_to_ndarrays(client_res.parameters)

        updated_weights = [
            (1 - self.learning_rate) * o + self.learning_rate * n
            for o, n in zip(old_weights, new_weights)
        ]
        self.parameters = ndarrays_to_parameters(updated_weights)

        self.learning_rate *= self.decay
        self.round += 1

        return self.parameters, {}  # No metrics here


    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is not None:
            return self.evaluate_fn(server_round, parameters)
        return None

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(num_clients=1)
        evaluate_ins = EvaluateIns(parameters=parameters, config={})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        global CURRENT_ROUND
        CURRENT_ROUND = server_round

        if not results:
            return None, {}

        # Format metrics for aggregation
        formatted = [
            (res.num_examples, res.metrics)
            for _, res in results
            if res.metrics and "f1_micro" in res.metrics  # avoid empty or invalid metrics
        ]

        # Compute weighted aggregated metrics
        if self.evaluate_metrics_aggregation_fn:
            aggregated = self.evaluate_metrics_aggregation_fn(formatted)
        else:
            aggregated = {}

        # Compute average loss
        total_examples = sum(res.num_examples for _, res in results)
        avg_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples

        return avg_loss, aggregated

