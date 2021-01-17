from argparse import ArgumentParser

import pytorch_lightning.callbacks as callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import yaml

from dialogpt import HuggingFaceModel
from load_data import HuggingFaceDataModule
import utils

import logging
logging.basicConfig(level=logging.INFO, format=utils.LOGGER_FORMAT)
logger = logging.getLogger(__file__)

def parse_callbacks(callback_config):
    callback_list = []
    logger.info("Loading the following callback config: %s", callback_config)
    for callback_info in callback_config:
        callback_class = getattr(callbacks, callback_info)
        callback_list.append(callback_class(**callback_config[callback_info]))
    return callback_list

def get_options():
    psr = ArgumentParser()
    psr.add_argument("--config-file", required=True, type=str, help="YAML configuration file for experiment")
    psr.add_argument("--test", action='store_true', help="Run evaluation on val set only.")
    psr.add_argument("--dry-run", action='store_true', help="If true, no logger will be used.")
    psr = Trainer.add_argparse_args(psr)
    return psr

if __name__ == '__main__':
    # get config
    psr = get_options()
    args = psr.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load data, model
    model = HuggingFaceModel(config["name"], config["model"])
    data = HuggingFaceDataModule(config["data"])

    # load experimental setup stuff
    callbacks = parse_callbacks(config.get("callbacks", {}))

    if not args.fast_dev_run and not args.dry_run:
        experiment_logger = WandbLogger(**config.get("logger", {}))
        experiment_logger.log_hyperparams(config["model"])
    else:
        experiment_logger = None
    trainer = Trainer.from_argparse_args(args, logger=experiment_logger, callbacks=callbacks)

    # run experiment
    if not args.test:
        data.setup('train')
        data.attach_special_tokens(model.model)
        trainer.fit(model, data.train_dataloader(), data.val_dataloader())
    else:
        data.setup('test')
        data.attach_special_tokens(model.model)
        logger.warning("To protect you from yourself, this eval loop loads the validation dataset. Change this manually if you want to actually run on test.")
        trainer.test(test_dataloaders=data.val_dataloader())
    if experiment_logger:
        experiment_logger.finalize()
