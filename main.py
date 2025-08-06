import argparse
import torch
import numpy as np
import multiprocessing
import flwr as fl
from flwr.server import ServerConfig

''' Project Imports '''
from model import FLModel
from utils import (
    load_and_preprocess,
    split_data_iid,
    create_weak_label_skew,
    create_medium_label_skew,
    create_strong_label_skew,
)
from client import FlowerClient
from server import strategy_sync, strategy_async, strategy_hybrid, set_experiment_context, set_current_round
from config import NUM_CLIENTS
from utils import set_seed

set_seed(42)  # REPRODUCIBILOTY !!!!!


def get_client_datasets(data_split: str, num_clients: int):
    X_train, X_test, y_train, y_test = load_and_preprocess()

    if data_split == "iid":
        return split_data_iid(X_train, y_train, num_clients), X_test, y_test
    elif data_split == "non-iid-weak":
        return create_weak_label_skew(X_train, y_train, num_clients), X_test, y_test
    elif data_split == "non-iid-medium":
        return create_medium_label_skew(X_train, y_train, num_clients), X_test, y_test
    elif data_split == "non-iid-strong":
        return create_strong_label_skew(X_train, y_train, num_clients), X_test, y_test
    else:
        raise ValueError("Invalid data split type")


def select_strategy(strategy_name: str):
    if strategy_name == "sync":
        return strategy_sync
    elif strategy_name == "async":
        return strategy_async
    elif strategy_name == "hybrid":
        return strategy_hybrid
    else:
        raise ValueError("Invalid strategy name")


def build_client_fn(client_datasets, X_test, y_test):
    def client_fn(cid: str):
        cid = int(cid)
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        testset = torch.utils.data.TensorDataset(test_tensor, y_test_tensor)
        model = FLModel()
        return FlowerClient(model, client_datasets[cid], testset).to_client()

    return client_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS, help="Number of clients")
    parser.add_argument("--strategy", type=str, choices=["sync", "async", "hybrid"], default="sync")
    parser.add_argument("--data", type=str, choices=["iid", "non-iid-weak", "non-iid-medium", "non-iid-strong"], default="iid")
    parser.add_argument("--rounds", type=int, default=3, help="Number of training rounds")
    args = parser.parse_args()


    strategy_map = {
        "sync": strategy_sync,
        "async": strategy_async,
        "hybrid": strategy_hybrid,
    }

    log_filename = f"logs/{args.strategy}_{args.data}_{args.clients}clients_{args.rounds}rounds.txt"
    set_experiment_context(log_filename)

    def fit_config_fn(server_round: int):
        set_current_round(server_round)
        return {}
    
    strategy = strategy_map[args.strategy]
    strategy.fit_config = fit_config_fn

    client_datasets, X_test, y_test = get_client_datasets(args.data, args.clients)
    strategy = select_strategy(args.strategy)
    client_fn = build_client_fn(client_datasets, X_test, y_test)

    multiprocessing.set_start_method("spawn")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        client_resources={"num_cpus": 1},
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
