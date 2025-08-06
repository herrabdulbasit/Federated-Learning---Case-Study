import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import FLModel
from sklearn.metrics import f1_score
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE
import numpy as np
from config import RANDOM_STATE

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainset, testset):
        self.model = model
        self.trainset = trainset
        self.testset = testset

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        train_loader = DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_STATE), num_workers=0)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        for _ in range(EPOCHS):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.trainset), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        test_loader = DataLoader(self.testset, batch_size=BATCH_SIZE, generator=torch.Generator().manual_seed(RANDOM_STATE), num_workers=0)
        criterion = torch.nn.BCELoss()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                preds = self.model(xb)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        preds_binary = (np.array(all_preds) >= 0.5).astype(int)
        labels = np.array(all_labels).astype(int)

        all_preds_np = np.array(all_preds).astype(np.float32)
        all_labels_np = np.array(all_labels).astype(np.float32)

        loss = criterion(torch.from_numpy(all_preds_np), torch.from_numpy(all_labels_np)).item()


        # Calculate F1 scores
        f1_micro = f1_score(labels, preds_binary, average="micro")
        f1_macro = f1_score(labels, preds_binary, average="macro")
        f1_class_0 = f1_score(labels, preds_binary, pos_label=0)
        f1_class_1 = f1_score(labels, preds_binary, pos_label=1)

        return loss, len(self.testset), {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_class_0": f1_class_0,
            "f1_class_1": f1_class_1,
        }