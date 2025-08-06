import flwr as fl
import numpy as np
import os
import torch
from flwr.common import ndarrays_to_parameters
from model import FLModel
from async_st import FedAsync

EXPERIMENT_LOG_PATH = None
CURRENT_ROUND = 0

def set_experiment_context(log_path: str):
    global EXPERIMENT_LOG_PATH
    EXPERIMENT_LOG_PATH = log_path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Clear previous log if exists
    with open(EXPERIMENT_LOG_PATH, "w") as f:
        f.write("")

def set_current_round(round_num: int):
    global CURRENT_ROUND
    CURRENT_ROUND = round_num

def weighted_average(metrics):
    print("Metrics:", metrics)
    total_examples = sum([num_examples for num_examples, _ in metrics])

    def avg(key):
        return sum([
            num_examples * client_metrics[key]
            for num_examples, client_metrics in metrics
        ]) / total_examples

    aggregated =  {
        "f1_micro": avg("f1_micro"),
        "f1_macro": avg("f1_macro"),
        "f1_class_0": avg("f1_class_0"),
        "f1_class_1": avg("f1_class_1"),
    }

    if EXPERIMENT_LOG_PATH:
        with open(EXPERIMENT_LOG_PATH, "a") as f:
            f.write(f"[Round {CURRENT_ROUND}]\n")
            for k, v in aggregated.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")

    return aggregated


class HybridStrategy(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit: float = 0.6, min_fit_clients: int = 2, **kwargs):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            **kwargs 
        )


strategy_sync = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)



model = FLModel()


def get_initial_parameters(model):
    # Convert model weights to list of numpy arrays
    weights = [val.cpu().numpy() for val in model.state_dict().values()]
    return ndarrays_to_parameters(weights)

initial_parameters = get_initial_parameters(model)

strategy_async = FedAsync(
    initial_parameters=initial_parameters,
    learning_rate=0.2,
    decay=0.99,
    evaluate_metrics_aggregation_fn=weighted_average
)

strategy_hybrid = HybridStrategy(
    fraction_fit=0.6,
    min_fit_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,  # passed once
)