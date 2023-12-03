import tempfile

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler

from src.conf import EPOCHS, TRAIN_SIZE, BATCH_SIZE
from src.data import Dataset
from src.networkmodule import NetworkModule
from src.plotter import create_gif, save_emb_plots


def save_pred(results, label, name):
    emb_df = pd.DataFrame(data=np.concatenate(results).reshape(-1, 2), columns=["emb_1", "emb_2"])
    emb_df["label"] = label.values
    emb_df.to_csv(f"data/{name}_result.csv", index=False)


def main():
    data = pd.read_csv("data/data.csv")
    target = pd.read_csv("data/labels.csv")

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=TRAIN_SIZE, shuffle=True,
                                                        stratify=target)

    train_dataset = Dataset(x_train, y_train)
    test_dataset = Dataset(x_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                   persistent_workers=True,
                                                   num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  persistent_workers=True,
                                                  num_workers=8)

    network_module = NetworkModule()

    logger = CSVLogger("logs/train_log")

    trainer = Trainer(accelerator="cpu", log_every_n_steps=1, max_epochs=EPOCHS, logger=logger)

    trainer.fit(model=network_module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    train_pred = trainer.predict(model=network_module, dataloaders=train_dataloader)
    test_pred = trainer.predict(model=network_module, dataloaders=test_dataloader)

    with tempfile.TemporaryDirectory() as dir:
        save_emb_plots(out_path=dir, network_module=network_module)
        create_gif(in_path=dir, out_path="plots")

    save_pred(train_pred, train_dataset.target, "train")
    save_pred(test_pred, test_dataset.target, "test")

    print("Finished the training process")


if __name__ == '__main__':
    main()
