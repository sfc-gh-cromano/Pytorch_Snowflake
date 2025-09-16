import os
import traceback
import warnings
from numbers import Number
from typing import List, Optional

import hydra
import lightning as L
import omegaconf
from loguru import logger

from tfh_train import utils


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "configs"), config_name="train")
def train(configs: omegaconf.DictConfig) -> Optional[Number]:
    """Training command.

    Args:
        configs (omegaconf.DictConfig): Training configuration.

    Returns:
        Optional[Number]: `config.optimized_metric` value used in when hyperparameters search with Optuna is performed.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    utils.logging.setup_loguru(function_name="train")

    try:
        logger.info("🚀 Training process started.")

        if configs.seed is not None:
            L.seed_everything(configs.seed, workers=True)

        logger.info("📝 Initializing loggers.")
        train_loggers: List[L.pytorch.loggers.logger.Logger] = utils.hydra_common.instantiate_list(configs.loggers)

        logger.info("📲 Initializing callbacks.")
        train_callbacks: List[L.Callback] = utils.hydra_common.instantiate_list(configs.callbacks)

        logger.info(f"📚 Instantiating lightning data module <{configs.data_module}>.")
        data_module: L.LightningDataModule = hydra.utils.instantiate(configs.data_module)

        logger.info(f"🕸 Instantiating lightning model module <{configs.model_module}>.")
        model_module: L.LightningModule = hydra.utils.instantiate(configs.model_module)

        logger.info(f"🐲 Instantiating lightning trainer <{configs.trainer}>.")
        trainer: L.Trainer = hydra.utils.instantiate(configs.trainer, callbacks=train_callbacks, logger=train_loggers)

        logger.info("🐉 Logging training essential parameters.")
        utils.logging.log_experiment_parameters({"configs": configs, "model_module": model_module, "trainer": trainer})

        logger.info("🔄 Starting training.")
        trainer.fit(model_module, data_module, ckpt_path=configs.ckpt_path)

        if optimized_metric := configs.get("optimized_metric"):
            return trainer.callback_metrics[optimized_metric].item()
    except Exception as e:
        logger.error(f"❌ Training processed failed with an exception {type(e).__name__}: {str(e)}")
        logger.error(f"❌ Error stack trace:\n {traceback.format_exc()}")
    finally:
        logger.info("🏁 Training process finished.")
        try:
            logger.info("🐉 Logging experiment directory with all artefacts.")
            utils.logging.log_experiment_dir(train_loggers)
        except Exception as e:
            logger.error(
                f"❌ Logging experiment directory with all artefacts FAILED with an exception {type(e).__name__}: {str(e)}"
            )
    return None


if __name__ == "__main__":
    train()
