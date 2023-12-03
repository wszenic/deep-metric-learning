import pandas as pd
import torch


class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.target = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item) -> tuple:
        return (
            torch.tensor(self.data.iloc[item, :].values, dtype=torch.float32),
            torch.tensor(self.target.iloc[item, :].values, dtype=torch.float32)
        )
