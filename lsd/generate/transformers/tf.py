from lsd import Base


class Transformer(Base):
    def __init__(self, params: dict):
        super().__init__(params)
        raise NotImplementedError(
            "Transformer Generator not implemented–coming soon!."
        )

    def setup(self):
        pass

    def train(self):
        pass

    def generate(self):
        pass
