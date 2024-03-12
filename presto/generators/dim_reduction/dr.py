from . import Config, GeneratorModule


class DimReductionModule(GeneratorModule):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)

    def setup(self):
        "process and unpack the parameters into their proper configs."
        # Find proper models and datasets
        # write proper config files
        pass

    def train(self):
        pass

    def generate(self):
        "Use a pretrained model to generate a latent space."
        pass
