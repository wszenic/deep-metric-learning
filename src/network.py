import torch
import torch.nn as nn
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.BatchNorm1d(30),
            nn.Linear(30, 20, bias=False),
            nn.Linear(20, 10, bias=False),
            nn.Linear(10, 2, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return x
