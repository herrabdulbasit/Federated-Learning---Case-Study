import torch.nn as nn

class FLModel(nn.Module):
    def __init__(self):
        super(FLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)