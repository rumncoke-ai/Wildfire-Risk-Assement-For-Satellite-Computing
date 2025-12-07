import logging
import os
import gc
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    del datamodule
    del trainer
    del model
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
    gc.collect()


# import warnings
# from importlib.util import find_spec
# from typing import Any, Callable, Dict, Optional, Tuple

# from omegaconf import DictConfig

# from src.utils import pylogger, rich_utils

# log = pylogger.RankedLogger(__name__, rank_zero_only=True)


# def extras(cfg: DictConfig) -> None:
#     """Applies optional utilities before the task is started.

#     Utilities:
#         - Ignoring python warnings
#         - Setting tags from command line
#         - Rich config printing

#     :param cfg: A DictConfig object containing the config tree.
#     """
#     # return if no `extras` config
#     if not cfg.get("extras"):
#         log.warning("Extras config not found! <cfg.extras=null>")
#         return

#     # disable python warnings
#     if cfg.extras.get("ignore_warnings"):
#         log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
#         warnings.filterwarnings("ignore")

#     # prompt user to input tags from command line if none are provided in the config
#     if cfg.extras.get("enforce_tags"):
#         log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
#         rich_utils.enforce_tags(cfg, save_to_file=True)

#     # pretty print config tree using Rich library
#     if cfg.extras.get("print_config"):
#         log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
#         rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


# def task_wrapper(task_func: Callable) -> Callable:
#     """Optional decorator that controls the failure behavior when executing the task function.

#     This wrapper can be used to:
#         - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
#         - save the exception to a `.log` file
#         - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
#         - etc. (adjust depending on your needs)

#     Example:
#     ```
#     @utils.task_wrapper
#     def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         ...
#         return metric_dict, object_dict
#     ```

#     :param task_func: The task function to be wrapped.

#     :return: The wrapped task function.
#     """

#     def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         # execute the task
#         try:
#             metric_dict, object_dict = task_func(cfg=cfg)

#         # things to do if exception occurs
#         except Exception as ex:
#             # save exception to `.log` file
#             log.exception("")

#             # some hyperparameter combinations might be invalid or cause out-of-memory errors
#             # so when using hparam search plugins like Optuna, you might want to disable
#             # raising the below exception to avoid multirun failure
#             raise ex

#         # things to always do after either success or exception
#         finally:
#             # display output dir path in terminal
#             log.info(f"Output dir: {cfg.paths.output_dir}")

#             # always close wandb run (even if exception occurs so multirun won't fail)
#             if find_spec("wandb"):  # check if wandb is installed
#                 import wandb

#                 if wandb.run:
#                     log.info("Closing wandb!")
#                     wandb.finish()

#         return metric_dict, object_dict

#     return wrap


# def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
#     """Safely retrieves value of the metric logged in LightningModule.

#     :param metric_dict: A dict containing metric values.
#     :param metric_name: If provided, the name of the metric to retrieve.
#     :return: If a metric name was provided, the value of the metric.
#     """
#     if not metric_name:
#         log.info("Metric name is None! Skipping metric value retrieval...")
#         return None

#     if metric_name not in metric_dict:
#         raise Exception(
#             f"Metric value not found! <metric_name={metric_name}>\n"
#             "Make sure metric name logged in LightningModule is correct!\n"
#             "Make sure `optimized_metric` name in `hparams_search` config is correct!"
#         )

#     metric_value = metric_dict[metric_name].item()
#     log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

#     return metric_value
