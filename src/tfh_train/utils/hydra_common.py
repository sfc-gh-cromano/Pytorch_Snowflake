from typing import Any, List, Optional

import hydra
import omegaconf
from loguru import logger


def get_output_directory() -> str:
    """Get `hydra` process output directory.

    Returns:
        str: `hydra` process output directory.
    """
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    return hydra_cfg["runtime"]["output_dir"]


def instantiate_list(objects_configs: Optional[omegaconf.DictConfig]) -> List[Any]:
    """Instantiate list of objects from config.

    Args:
        objects_configs (Optional[omegaconf.DictConfig]): Objects configs.

    Returns:
        List[Any]: Instantiated objects
    """
    loggers: List[Any] = []

    if not objects_configs:
        logger.warning("🫙 No objects declarations found in configs dictionary!")
        return loggers

    for _, obj_config in objects_configs.items():
        if isinstance(obj_config, omegaconf.DictConfig) and "_target_" in obj_config:
            logger.info(f"🚜 Instantiating object <{obj_config}>")
            loggers.append(hydra.utils.instantiate(obj_config))

    return loggers
