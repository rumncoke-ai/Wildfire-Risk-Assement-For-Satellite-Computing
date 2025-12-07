from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from wildfire_forecasting.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing with best model!")
        trainer.test(datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")

    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


# from typing import Any, Dict, List, Optional, Tuple

# import hydra
# import lightning as L
# import rootutils
# import torch
# from lightning import Callback, LightningDataModule, LightningModule, Trainer
# from lightning.pytorch.loggers import Logger
# from omegaconf import DictConfig

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# # ------------------------------------------------------------------------------------ #
# # the setup_root above is equivalent to:
# # - adding project root dir to PYTHONPATH
# #       (so you don't need to force user to install project as a package)
# #       (necessary before importing any local modules e.g. `from src import utils`)
# # - setting up PROJECT_ROOT environment variable
# #       (which is used as a base for paths in "configs/paths/default.yaml")
# #       (this way all filepaths are the same no matter where you run the code)
# # - loading environment variables from ".env" in root dir
# #
# # you can remove it if you:
# # 1. either install project as a package or move entry files to project root dir
# # 2. set `root_dir` to "." in "configs/paths/default.yaml"
# #
# # more info: https://github.com/ashleve/rootutils
# # ------------------------------------------------------------------------------------ #

# from src.utils import (
#     RankedLogger,
#     extras,
#     get_metric_value,
#     instantiate_callbacks,
#     instantiate_loggers,
#     log_hyperparameters,
#     task_wrapper,
# )

# log = RankedLogger(__name__, rank_zero_only=True)


# @task_wrapper
# def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
#     training.

#     This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
#     failure. Useful for multiruns, saving info about the crash, etc.

#     :param cfg: A DictConfig configuration composed by Hydra.
#     :return: A tuple with metrics and dict with all instantiated objects.
#     """
#     # set seed for random number generators in pytorch, numpy and python.random
#     if cfg.get("seed"):
#         L.seed_everything(cfg.seed, workers=True)

#     log.info(f"Instantiating datamodule <{cfg.data._target_}>")
#     datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

#     log.info(f"Instantiating model <{cfg.model._target_}>")
#     model: LightningModule = hydra.utils.instantiate(cfg.model)

#     log.info("Instantiating callbacks...")
#     callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

#     log.info("Instantiating loggers...")
#     logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

#     log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
#     trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

#     object_dict = {
#         "cfg": cfg,
#         "datamodule": datamodule,
#         "model": model,
#         "callbacks": callbacks,
#         "logger": logger,
#         "trainer": trainer,
#     }

#     if logger:
#         log.info("Logging hyperparameters!")
#         log_hyperparameters(object_dict)

#     if cfg.get("train"):
#         log.info("Starting training!")
#         trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

#     train_metrics = trainer.callback_metrics

#     if cfg.get("test"):
#         log.info("Starting testing!")
#         ckpt_path = trainer.checkpoint_callback.best_model_path
#         if ckpt_path == "":
#             log.warning("Best ckpt not found! Using current weights for testing...")
#             ckpt_path = None
#         trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
#         log.info(f"Best ckpt path: {ckpt_path}")

#     test_metrics = trainer.callback_metrics

#     # merge train and test metrics
#     metric_dict = {**train_metrics, **test_metrics}

#     return metric_dict, object_dict


# @hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
# def main(cfg: DictConfig) -> Optional[float]:
#     """Main entry point for training.

#     :param cfg: DictConfig configuration composed by Hydra.
#     :return: Optional[float] with optimized metric value.
#     """
#     # apply extra utilities
#     # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
#     extras(cfg)

#     # train the model
#     metric_dict, _ = train(cfg)

#     # safely retrieve metric value for hydra-based hyperparameter optimization
#     metric_value = get_metric_value(
#         metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
#     )

#     # return optimized metric
#     return metric_value


# if __name__ == "__main__":
#     main()


