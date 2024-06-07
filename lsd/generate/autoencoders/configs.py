from dataclasses import dataclass

#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class celebA:
    name = "celebA TEST"


@dataclass
class MNIST:
    name = "MNIST TEST"


#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯
@dataclass
class betaVAE:
    name = "betaVAE TEST"


@dataclass
class infoVAE:
    name = "infoVAE TEST"


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Adam:
    name = "Adam TEST"
