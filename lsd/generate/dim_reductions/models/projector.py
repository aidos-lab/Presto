from abc import ABC, abstractmethod
from lsd.generate.dim_reductions.configs import Projector


class Projector(ABC):
    def __init__(self, config: Projector):
        self.config = config

    @property
    def dim(self):
        return self.config.dim

    @property
    def metric(self):
        return self.config.dim

    @abstractmethod
    def project(self, data):
        raise NotImplementedError()
