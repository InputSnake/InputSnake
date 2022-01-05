from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import models
from fiddle_data_module import MimicDataModule

model_map = {
    "MortCnn": models.MortCnn,
    "DiagCnn": models.DiagCnn,
    "LosCnn": models.LosCnn,
    "InputSnake": models.LosInputSnake,
    "MortInputSnake": models.MortInputSnake,
    "OtherStrategies": models.OtherStrategies
}


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="Project name")
    parser.add_argument("--model", type=str, default="Model name")
    parser.add_argument("--name", type=str, default="Run name")
    parser.add_argument("--log_dir", type=str, default="/path/to/logs")
    parser.add_argument("--mode", type=str, default="max")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MimicDataModule.add_argparse_args(parser)
    temp_args, _ = parser.parse_known_args()
    parser = model_map[temp_args.model].add_argparse_args(parser)
    return parser.parse_args(args)


def main(args):
    dm = MimicDataModule.from_argparse_args(args)
    model = model_map[args.model](**vars(args))
    print(f"Run {args.model}")
    wandb_logger = WandbLogger(project=args.project, name=args.name, log_model=False, save_dir=args.log_dir)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, accelerator="gpu", gpus=[3], max_epochs=15,
                                            callbacks=[ModelCheckpoint(monitor="score", mode=args.mode)])

    trainer.fit(model, datamodule=dm)

    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
