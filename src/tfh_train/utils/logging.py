import os
import sys
from typing import Any, Dict, List

import lightning as L
import mlflow
from loguru import logger

from tfh_train.utils.hydra_common import get_output_directory


def setup_loguru(function_name: str) -> None:
    """Set command's loguru.logger.

    Args:
        function_name (str): Function/command name.
    """
    logger.remove()
    logger.add(os.path.join(get_output_directory(), f"{function_name}.log"))
    logger.add(sys.stdout, colorize=True, level="DEBUG")


@L.pytorch.utilities.rank_zero_only
def log_experiment_parameters(parameters2log: Dict[str, Any]) -> None:
    """Log command parameters.

    Args:
        parameters2log (Dict[str, Any]): Dictionary with parameters and hyperparameters to log.
    """
    hparams2log: Dict[str, Any] = {}

    configs = parameters2log["configs"]
    model = parameters2log["model_module"]
    trainer = parameters2log["trainer"]

    hparams2log["model_module"] = configs.get("model_module")

    hparams2log["model.params.total"] = sum(p.numel() for p in model.parameters())
    hparams2log["model.params.trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams2log["model.params.non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams2log["data_module"] = configs.get("data_module")
    hparams2log["trainer"] = configs.get("trainer")

    hparams2log["callbacks"] = configs.get("callbacks")

    hparams2log["command_name"] = configs.get("command_name")
    hparams2log["tags"] = configs.get("tags")
    hparams2log["ckpt_path"] = configs.get("ckpt_path")
    hparams2log["seed"] = configs.get("seed")

    for command_logger in trainer.loggers:
        command_logger.log_hyperparams(hparams2log)


@L.pytorch.utilities.rank_zero_only
def log_experiment_dir(loggers: List[L.pytorch.loggers.logger.Logger]) -> None:
    """Log experiment directory to loggers of specific type.

        List of support loggers:
            - lightning.pytorch.loggers.mlflow.MLFlowLogger

    Args:
        loggers (List[L.pytorch.loggers.logger.Logger]): List of loggers used during command.
    """
    for command_logger in loggers:
        if isinstance(command_logger, L.pytorch.loggers.mlflow.MLFlowLogger):
            mlflow.set_tracking_uri(command_logger.experiment.tracking_uri)
            command_logger.experiment.log_artifacts(command_logger.run_id, local_dir=get_output_directory())
