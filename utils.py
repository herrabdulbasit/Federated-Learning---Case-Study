import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from config import NUM_CLIENTS, RANDOM_STATE
import random
import os

def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

def load_and_preprocess():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(["Class", "Time"], axis=1).values
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def split_data_iid(X_train, y_train, num_cl=NUM_CLIENTS):
    client_datasets = []
    X_splits = np.array_split(X_train, NUM_CLIENTS)
    y_splits = np.array_split(y_train, NUM_CLIENTS)
    for i in range(NUM_CLIENTS):
        X_client = torch.tensor(X_splits[i], dtype=torch.float32)
        y_client = torch.tensor(y_splits[i], dtype=torch.float32).unsqueeze(1)
        client_datasets.append(TensorDataset(X_client, y_client))
    return client_datasets

def create_weak_label_skew(X_np, y_np, num_clients=NUM_CLIENTS, min_frauds_per_client=5):
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset

    fraud_ratios = [0.001, 0.003, 0.005, 0.008, 0.01]
    assert len(fraud_ratios) == num_clients

    total_samples = len(y_np)
    total_ratio = np.array(fraud_ratios) / sum(fraud_ratios)
    client_sample_counts = (total_ratio * total_samples).astype(int)
    client_sample_counts[-1] = total_samples - client_sample_counts[:-1].sum()

    fraud_idx = np.where(y_np == 1)[0]
    normal_idx = np.where(y_np == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(fraud_idx)
    rng.shuffle(normal_idx)

    total_frauds = len(fraud_idx)
    total_normals = len(normal_idx)

    # Step 1: assign minimum frauds
    guaranteed_frauds = min_frauds_per_client * num_clients
    if guaranteed_frauds > total_frauds:
        raise ValueError("Not enough frauds to assign minimum per client")

    fraud_counts = [min_frauds_per_client] * num_clients
    remaining_frauds = total_frauds - guaranteed_frauds

    # Step 2: distribute remaining frauds according to target skew
    fraud_weights = np.array(fraud_ratios) / sum(fraud_ratios)
    extra_frauds = (fraud_weights * remaining_frauds).astype(int)
    extra_frauds[-1] = remaining_frauds - sum(extra_frauds[:-1])

    fraud_counts = [base + extra for base, extra in zip(fraud_counts, extra_frauds)]

    # Step 3: assign normal samples to match target total size
    normal_counts = [max(client_sample_counts[i] - fraud_counts[i], 0) for i in range(num_clients)]

    # Build datasets
    fraud_used, normal_used = 0, 0
    client_datasets = []

    for i in range(num_clients):
        f_count = fraud_counts[i]
        n_count = normal_counts[i]

        fraud_part = fraud_idx[fraud_used:fraud_used + f_count]
        normal_part = normal_idx[normal_used:normal_used + n_count]

        fraud_used += f_count
        normal_used += n_count

        indices = np.concatenate([fraud_part, normal_part])
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(indices)

        X_client = torch.tensor(X_np[indices], dtype=torch.float32)

        y_raw = y_np[indices]
        #Ensure label shape is [N, 1]
        if y_raw.ndim == 1:
            y_client = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
        elif y_raw.ndim == 2 and y_raw.shape[1] == 1:
            y_client = torch.tensor(y_raw, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected label shape: {y_raw.shape}")

        dataset = TensorDataset(X_client, y_client)
        client_datasets.append(dataset)

        class1 = int(y_client.sum().item())
        print(f"Client {i+1}: total={len(y_client)}, class 1={class1}, ratio={class1 / len(y_client):.4f}")

    print(f"\nTotal frauds used: {fraud_used} / {total_frauds}")
    print(f"Total normals used: {normal_used} / {total_normals}")
    print(f"Total samples used: {fraud_used + normal_used} / {len(y_np)}")

    return client_datasets

def create_medium_label_skew(X_np, y_np, num_clients=NUM_CLIENTS, min_frauds_per_client=10):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    # Define moderately skewed label ratios per client
    fraud_ratios = [0.001, 0.0025, 0.005, 0.0075, 0.01]  # medium skew
    assert len(fraud_ratios) == num_clients

    total_samples = len(y_np)
    total_ratio = np.array(fraud_ratios) / sum(fraud_ratios)
    client_sample_counts = (total_ratio * total_samples).astype(int)
    client_sample_counts[-1] = total_samples - client_sample_counts[:-1].sum()

    fraud_idx = np.where(y_np == 1)[0]
    normal_idx = np.where(y_np == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(fraud_idx)
    rng.shuffle(normal_idx)

    total_frauds = len(fraud_idx)
    total_normals = len(normal_idx)

    guaranteed_frauds = min_frauds_per_client * num_clients
    if guaranteed_frauds > total_frauds:
        raise ValueError("Not enough frauds to assign minimum per client")

    fraud_counts = [min_frauds_per_client] * num_clients
    remaining_frauds = total_frauds - guaranteed_frauds

    fraud_weights = np.array(fraud_ratios) / sum(fraud_ratios)
    extra_frauds = (fraud_weights * remaining_frauds).astype(int)
    extra_frauds[-1] = remaining_frauds - sum(extra_frauds[:-1])
    fraud_counts = [base + extra for base, extra in zip(fraud_counts, extra_frauds)]

    normal_counts = [max(client_sample_counts[i] - fraud_counts[i], 0) for i in range(num_clients)]

    fraud_used, normal_used = 0, 0
    client_datasets = []

    for i in range(num_clients):
        f_count = fraud_counts[i]
        n_count = normal_counts[i]

        fraud_part = fraud_idx[fraud_used:fraud_used + f_count]
        normal_part = normal_idx[normal_used:normal_used + n_count]

        fraud_used += f_count
        normal_used += n_count

        indices = np.concatenate([fraud_part, normal_part])
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(indices)

        X_client = torch.tensor(X_np[indices], dtype=torch.float32)

        # Safely fix label shape to [N, 1]
        y_raw = y_np[indices]
        if y_raw.ndim == 1:
            y_client = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
        elif y_raw.ndim == 2 and y_raw.shape[1] == 1:
            y_client = torch.tensor(y_raw, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected label shape: {y_raw.shape}")

        dataset = TensorDataset(X_client, y_client)
        client_datasets.append(dataset)

        class1 = int(y_client.sum().item())
        print(f"Client {i+1}: total={len(y_client)}, class 1={class1}, ratio={class1 / len(y_client):.4f}")

    print(f"\nTotal frauds used: {fraud_used} / {total_frauds}")
    print(f"Total normals used: {normal_used} / {total_normals}")
    print(f"Total samples used: {fraud_used + normal_used} / {len(y_np)}")

    return client_datasets



def create_strong_label_skew(X_np, y_np, num_clients=NUM_CLIENTS, fraud_client_ids=[1, 3]):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    fraud_idx = np.where(y_np == 1)[0]
    normal_idx = np.where(y_np == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)  
    rng.shuffle(fraud_idx)
    rng.shuffle(normal_idx)

    total_frauds = len(fraud_idx)
    total_normals = len(normal_idx)

    # Divide samples across clients unevenly
    normal_splits = np.array_split(normal_idx, num_clients)
    fraud_splits = np.array_split(fraud_idx, len(fraud_client_ids))

    client_datasets = []

    for i in range(num_clients):
        include_fraud = i in fraud_client_ids

        if include_fraud:
            fraud_split = fraud_splits[fraud_client_ids.index(i)]
        else:
            fraud_split = np.array([], dtype=int)

        normal_split = normal_splits[i]

        indices = np.concatenate([normal_split, fraud_split])
        rng = np.random.default_rng(RANDOM_STATE)           
        rng.shuffle(indices)

        X_client = torch.tensor(X_np[indices], dtype=torch.float32)
        y_raw = y_np[indices]

        # Clean label shape
        if y_raw.ndim == 1:
            y_client = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)
        elif y_raw.ndim == 2 and y_raw.shape[1] == 1:
            y_client = torch.tensor(y_raw, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected y shape: {y_raw.shape}")

        dataset = TensorDataset(X_client, y_client)
        client_datasets.append(dataset)

        class1 = int(y_client.sum().item())
        print(f"Client {i+1}: total={len(y_client)}, class 1={class1}, ratio={class1 / len(y_client):.4f}")

    print(f"\nStrong label skew distribution complete.")
    return client_datasets
