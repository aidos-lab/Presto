from lsd import Base


class Transformer(Base):
    def __init__(self, params: dict):
        super().__init__(params)

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


class Custom(Base):
    def __init__(self, params: dict):
        super().__init__(params)

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
