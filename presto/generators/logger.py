from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable

import wandb


class Logger:
    def __init__(
        self,
        exp: str,
        name: str,
        dev: bool = True,
        out_file: bool = True,
    ):
        self.dev = dev
        self.wandb = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if out_file:
            logs_dir = f"{root}/experiments/{exp}/logs/"
            if not os.path.isdir(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)

            log_file = os.path.join(logs_dir, f"{name}.log")
            fh = logging.FileHandler(filename=log_file, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def wandb_init(self, model, config):
        if self.dev:
            self.wandb = None
        else:
            self.wandb = wandb.init(
                project=config.meta.name,
                group=f"({config.data_params.batch_size})",
                name=f"({config.model_params.module})",
                tags=config.meta.tags,
            )
            wandb.watch(model, log_freq=100)

    def log(self, msg: str | None = None, params: dict[str, str] | None = None) -> None:
        if msg:
            self.logger.info(msg=msg)

        if params and self.wandb:
            self.wandb.log(params)

    def log_config(self, config):
        if self.wandb:
            self.wandb.config.update(config)
        else:
            raise ValueError()


def timing(logger: Logger, dev: bool = True):
    if dev:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.log(f"Calling {func.__name__}")
                ts = time.time()
                value = func(*args, **kwargs)
                te = time.time()
                logger.log(f"Finished {func.__name__}")
                if logger:
                    logger.log("func:%r took: %2.4f sec" % (func.__name__, te - ts))
                return value

            return wrapper

        return decorator
    else:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                value = func(*args, **kwargs)
                return value

            return wrapper

        return decorator
