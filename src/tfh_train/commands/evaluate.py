import os
import traceback
import warnings
from typing import List

import hydra
import lightning as L
import omegaconf
from loguru import logger

from tfh_train import utils


class TFHEvaluationCommandError(Exception):
    """Evaluation command exception class."""

    pass


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "configs"), config_name="evaluate")
def evaluate(configs: omegaconf.DictConfig) -> None:
    """Evaluate command.

    Args:
        configs (omegaconf.DictConfig): Evaluation configuration.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    utils.logging.setup_loguru(function_name="evaluate")

    try:
        if configs.ckpt_path is None:
            raise TFHEvaluationCommandError(f"{configs.ckpt_path=} parameter not specified in configs.")

        logger.info("🚀 Evaluation process started.")

        logger.info("📝 Initializing loggers.")
        eval_loggers: List[L.pytorch.loggers.logger.Logger] = utils.hydra_common.instantiate_list(configs.loggers)

        logger.info("📲 Initializing callbacks.")
        eval_callbacks: List[L.Callback] = utils.hydra_common.instantiate_list(configs.callbacks)

        logger.info(f"📚 Instantiating lightning data module <{configs.data_module}>.")
        data_module: L.LightningDataModule = hydra.utils.instantiate(configs.data_module)

        logger.info(f"🕸 Instantiating lightning model module <{configs.model_module}>.")
        model_module: L.LightningModule = hydra.utils.instantiate(configs.model_module)

        logger.info(f"🐲 Instantiating lightning trainer <{configs.trainer}>.")
        trainer: L.Trainer = hydra.utils.instantiate(configs.trainer, callbacks=eval_callbacks, logger=eval_loggers)

        logger.info("🐉 Logging evaluation essential parameters.")
        utils.logging.log_experiment_parameters({"configs": configs, "model_module": model_module, "trainer": trainer})

        logger.info("🔄 Starting evaluation.")
        trainer.test(model_module, data_module, ckpt_path=configs.ckpt_path)
    except Exception as e:
        logger.error(f"❌ Evaluation processed failed with an exception {type(e).__name__}: {str(e)}")
        logger.error(f"❌ error stack trace {traceback.format_exc()}")
    finally:
        logger.info("🏁 Evaluation process finished.")
        try:
            logger.info("🐉 Logging experiment directory with all artefacts.")
            utils.logging.log_experiment_dir(eval_loggers)
        except Exception:
            logger.error("❌ Logging experiment directory with all artefacts FAILED.")


if __name__ == "__main__":
    evaluate()
