from abc import ABC, abstractmethod
from lsd.generate.dim_reductions.configs import Projector
from inspect import signature


class BaseProjector(ABC):
    def __init__(self, config: Projector):
        self.config = config

    @property
    def dim(self):
        return self.config.n_components

    @property
    def metric(self):
        return self.config.metric

    @abstractmethod
    def project(self, data):
        raise NotImplementedError()

    @staticmethod
    def _get_parameters(Operator, config) -> dict:
        params = signature(Operator.__init__).parameters
        args = {
            name: config.get(name, param.default)
            for name, param in params.items()
            if param.default != param.empty
        }
        return args


def initialize():
    return BaseProjector
