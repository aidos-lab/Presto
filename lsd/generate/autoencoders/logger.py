from typing import Any, Callable, Optional, Dict
import functools
import logging
import os
import time
import wandb
from lsd.utils import get_wandb_env


class Logger:
    def __init__(
        self,
        exp: str,
        name: str,
        wandb_logging: bool = False,
        out_file: bool = True,
    ):
        self.wandb_logging = wandb_logging
        self.wandb = None
        # Set up the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        if out_file:
            logs_dir = os.path.join(exp, "logs/")
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, f"{name}.log")

            # File handler
            fh = logging.FileHandler(filename=log_file, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def wandb_init(self, model: Any) -> None:
        if self.wandb_logging:
            cfg = get_wandb_env()
            self.wandb = wandb.init(
                project=cfg["wandb_project"],
                entity=cfg["wandb_entity"],
                tags=[cfg["wandb_tag"]],
            )
            wandb.watch(model, log_freq=100)

    def log(
        self, msg: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ) -> None:
        if msg:
            self.logger.info(msg)
        if params and self.wandb:
            self.wandb.log(params)

    def log_config(self, config: Any) -> None:
        if self.wandb:
            self.wandb.config.update(config)
        else:
            raise ValueError("WandB logging is not initialized or disabled.")
