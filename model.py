import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import optuna


def activation(activ_type: str):
    if activ_type == "ReLU":
        return torch.nn.ReLU()
    elif activ_type == "LeakyReLU":
        return torch.nn.LeakyReLU()
    elif activ_type == "ELU":
        return torch.nn.ELU()
    else:
        raise Exception("Wrong activation function")


class CorrectionModel(pl.LightningModule):
    def __init__(self, trial: optuna.Trial):
        super().__init__()

        self.number_of_layers = trial.suggest_int("number_of_layers", 2, 6)

        layer_widths = []
        for i in range(0, self.number_of_layers):
            max_width = 500 if i > 3 else 1500
            layer_widths.append(trial.suggest_int(f"layer_{i}_width", 20, max_width))

        self.dropout = trial.suggest_uniform("dropout", 0.2, 0.8)
        self.activ_type = trial.suggest_categorical(
            "activ_type", ["ReLU", "ELU", "LeakyReLU"]
        )
        self.loss_fn = trial.suggest_categorical("loss_fn", ["mae", "mse"])
        self.lr = trial.suggest_loguniform("lr", 1e-8, 1e-2)
        self.alpha = trial.suggest_uniform("alpha", 0.8, 5.0)

        modules = []

        modules.append(
            torch.nn.Sequential(
                torch.nn.Linear(3, layer_widths[0]), activation(self.activ_type)
            )
        )

        for i in range(self.number_of_layers - 1):
            modules.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_widths[i], layer_widths[i + 1]),
                    activation(self.activ_type),
                    torch.nn.Dropout(self.dropout),
                )
            )

        modules.append(torch.nn.Sequential(torch.nn.Linear(layer_widths[-1], 2)))

        self.neural_network = torch.nn.Sequential(*modules)

        self.step = 0

    def forward(self, x):
        r = (x ** 2).sum(1).view(-1, 1) - 2
        x_and_r = torch.cat((x, r), dim=1)
        y_hat = self.neural_network(x_and_r)
        return y_hat

    def training_step(self, batch, batch_idx):
        est = batch["est"].float()
        diff = batch["diff"].float()

        diff_hat = self.forward(est)

        mse = F.mse_loss(diff_hat, diff)
        mae = F.l1_loss(diff_hat, diff)
        metrics = {"train_mse": mse.item(), "train_mae": mae.item()}

        self.logger.log_metrics(metrics, step=self.step)
        self.step += 1

        loss = mse if self.loss_fn == "mse" else mae

        return {"loss": loss, "train_mse": mse, "train_mae": mae}

    def validation_step(self, batch, batch_idx):
        est = batch["est"].float()
        diff = batch["diff"].float()

        diff_hat = self.forward(est)

        mse = F.mse_loss(diff_hat, diff)
        mae = F.l1_loss(diff_hat, diff)
        metrics = {"val_mse": mse.item(), "val_mae": mae.item()}

        self.logger.log_metrics(metrics, step=self.step)
        self.step += 1

        loss = mse if self.loss_fn == "mse" else mae

        return {"loss": loss, "val_mse": mse, "val_mae": mae}

    def test_step(self, batch, batch_idx):
        est = batch["est"].float()
        diff = batch["diff"].float()

        diff_hat = self.forward(est)

        mse = F.mse_loss(diff_hat, diff)
        mae = F.l1_loss(diff_hat, diff)
        metrics = {"test_mse": mse.item(), "test_mae": mae.item()}

        self.logger.log_metrics(metrics, step=self.step)
        self.step += 1

        loss = mse if self.loss_fn == "mse" else mae

        return {"loss": loss, "test_mse": mse, "test_mae": mae}

    def configure_optimizers(self):
        print(self.lr)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda epoch: 2 ** (-epoch * self.alpha), verbose=True
        )
        return [opt], [sch]
