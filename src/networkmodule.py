import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_metric_learning import miners, losses

from src.conf import LR, TRIPLET_MARGIN, TRIPLETS_PER_ANCHOR
from src.network import Network


class NetworkModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Network()

        self.miner = miners.BatchHardMiner()
        self.loss = losses.TripletMarginLoss(margin=TRIPLET_MARGIN, triplets_per_anchor=TRIPLETS_PER_ANCHOR, smooth_loss=True)

        self.data = []
        self.labels = []

        self.res = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        data, target = batch
        labels = target.reshape(-1)

        embeddings = self.model(data)
        pairs = self.miner(embeddings, labels)

        loss = self.loss(embeddings, labels, pairs)
        self.log(name="train_loss", value=loss)
        return loss

    def validation_step(self, batch):
        data, target = batch
        labels = target.reshape(-1)

        embeddings = self.model(data)
        pairs = self.miner(embeddings, labels)

        loss = self.loss(embeddings, labels, pairs)

        self.data.append(embeddings)
        self.labels.append(labels)

        self.log(name="val_loss", value=loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        embs = np.concatenate(self.data)
        labs = np.concatenate(self.labels)

        self.res.append(pd.DataFrame({
            "emb_0": embs[:, 0],
            "emb_1": embs[:, 1],
            "labels": labs
        }))

        self.data = []
        self.labels = []
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=LR)

    def predict_step(self, x):
        data, _ = x
        return self.model(data)

    def _dataset_step(self, batch, loss_name):
        data, target = batch
        labels = target.reshape(-1)

        embeddings = self.model(data)
        pairs = self.miner(embeddings, labels)

        loss = self.loss(embeddings, labels, pairs)
        self.log(name=loss_name, value=loss)
        return loss
