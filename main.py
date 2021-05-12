import os
import pandas as pd
import pickle
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from model import CorrectionModel
from dataset import create_train_test_dataloaders


def find_smallest_mae(run_name: str):
    """Returns the smallest MAE for a given run. Based on outputs of the CSVLogger."""
    version_folders = os.listdir(os.path.join("csv_logs", run_name))
    max_version = max([int(f[-1]) for f in version_folders])
    metrics = pd.read_csv(
        os.path.join("csv_logs", run_name, f"version_{max_version}", "metrics.csv")
    )
    return metrics["val_mae"].min()


def objective(trial: optuna.Trial):
    """Optuna will minimize the return value of this function."""

    run_name = "trial_{}".format(trial.number)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("models", run_name), monitor="val_loss", mode="min", save_last=True
    )

    tb_logger = TensorBoardLogger(save_dir="logs")
    csv_logger = CSVLogger("csv_logs", name=run_name)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        checkpoint_callback=True,
        val_check_interval=0.33,
        max_epochs=8,
        gpus=1,
        callbacks=checkpoint_callback,
        auto_lr_find=True,
    )

    train_dataloader, val_dataloader, _ = create_train_test_dataloaders(fake_examles=True)
    model = CorrectionModel(trial=trial)

    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except RuntimeError:
        return 1

    return find_smallest_mae(run_name)


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)

    study.enqueue_trial(
        {
            "number_of_layers": 3,
            "layer_0_width": 500,
            "layer_1_width": 50,
            "layer_2_width": 50,
            "dropout": 0.6,
            "activ_type": "ELU",
            "loss_fn": "mae",
            "lr": 0.00002,
            "alpha": 1.53,
        }
    )
    study.enqueue_trial(
        {
            "number_of_layers": 6,
            "layer_0_width": 1500,
            "layer_1_width": 1500,
            "layer_2_width": 1500,
            "layer_3_width": 134,
            "layer_4_width": 130,
            "layer_5_width": 373,
            "dropout": 0.3,
            "activ_type": "ELU",
            "loss_fn": "mse",
            "lr": 0.00012,
            "alpha": 3.95,
        }
    )
    study.enqueue_trial(
        {
            "number_of_layers": 3,
            "layer_0_width": 1500,
            "layer_1_width": 200,
            "layer_2_width": 200,
            "dropout": 0.6,
            "activ_type": "ELU",
            "loss_fn": "mae",
            "lr": 0.00002,
            "alpha": 1.838,
        }
    )
    study.enqueue_trial(
        {
            "number_of_layers": 4,
            "layer_0_width": 1030,
            "layer_1_width": 150,
            "layer_2_width": 1033,
            "layer_3_width": 468,
            "dropout": 0.5097128586459475,
            "activ_type": "ReLU",
            "loss_fn": "mae",
            "lr": 0.0003022651981276079,
            "alpha": 1.4326179329364277,
        }
    )

    study.optimize(objective, n_trials=150, timeout=60 * 60 * 8)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("\tValue: {}".format(best_trial.value))

    print("\tParams: ")

    for key, value in best_trial.params.items():
        print("\t{}: {}".format(key, value))

    with open("study.pkl", "wb") as f:
        pickle.dump(study, f)

    print("Dumped study.")