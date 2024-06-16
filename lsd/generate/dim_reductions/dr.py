import os
import importlib
from sklearn import datasets as sc
import omegaconf

from lsd import Base
from lsd.utils import extract_yaml_id


class DimReduction(Base):
    def __init__(self, params: dict):
        super().__init__(params)
        self.projector_cfg = self.setup()

    def setup(self):

        projector_cfg = omegaconf.OmegaConf.create({})
        projector_cfg.experiment = self.params.experiment
        projector_cfg.id = extract_yaml_id(self.params.file)

        projector_cfg.model = self.params.model_choices.get("module", "")

        for _, sub_dict in self.params.items():
            if isinstance(sub_dict, omegaconf.dictconfig.DictConfig):
                for key, value in sub_dict.items():
                    if key not in [
                        "module",
                        "name",
                    ]:  # Exclude already processed keys
                        projector_cfg[key] = value

        # Data
        if self.params.data_choices.generator:
            if "module" in self.params.data_choices.keys():
                projector_cfg.path = self.params.data_choices.path
                dm = importlib.import_module(self.params.data_choices.module)
                loader = getattr(dm, self.params.data_choices.generator)
                self.data, self.labels = loader(**self.params.data_choices)

            else:
                print("Loading sklearn data")
                loader = getattr(sc, self.params.data_choices.generator)
                self.data, self.labels = loader(
                    return_X_y=True,
                )

        self.model = importlib.import_module(projector_cfg.model).initialize()
        self.outDir = os.path.join(projector_cfg.experiment, "latent_spaces/")
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)

        self.outFile = os.path.join(
            self.outDir, f"universe_{projector_cfg.id}.pkl"
        )
        return projector_cfg

    def train(self):
        pass

    def generate(self):
        "Use a pretrained model to generate a latent space."
        model = self.model(self.projector_cfg)
        L = model.project(self.data)

        self.write_pkl(L, self.outFile)

        del model, L
