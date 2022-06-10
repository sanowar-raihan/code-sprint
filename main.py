import argparse
import yaml
import pandas as pd
import seaborn as sn
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from preprocess.transforms import train_transforms, test_transforms
from models.litresnet import LitResnet


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help='path to the config file')
    args = parser.parse_args()

    with open(args.config_path) as config_file:
        cfg = yaml.load(config_file, Loader=yaml.Fullloader)

    seed_everything(cfg.exp_params.seed)

    cifar10_dm = CIFAR10DataModule(
        data_dir=cfg.exp_params.data_dir,
        batch_size=cfg.exp_params.batch_size,
        num_workers=cfg.exp_params.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = LitResnet(
        lr=cfg.exp_params.lr,
        momentum=cfg.exp_params.momentum,
        weight_decay=cfg.exp_params.weight_decay,
        batch_size=cfg.exp_params.batch_size,
    )

    trainer = Trainer(
        max_epochs=cfg.exp_params.max_epoch,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir=cfg.log_params.save_dir),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)


    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    sn.relplot(data=metrics, kind="line")