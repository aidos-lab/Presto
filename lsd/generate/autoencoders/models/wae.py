from torch.nn import functional as F

from lsd.generate.autoencoders.models.vae import BaseVAE


class WAE(BaseVAE):
    def __init__(
        self,
        config,
        **kwargs,
    ) -> None:
        super(WAE, self).__init__(config)

        self.kernel_type = config.kernel
        self.reg_weight = config.reg_weight
        self.z_var = config.z_var

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss = F.mse_loss(recons, input)

        mmd_loss = self.compute_mmd(
            z,
            self.kernel_type,
            reg_weight,
            self.z_var,
            self.eps,
        )

        loss = recons_loss + mmd_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "MMD": mmd_loss,
        }


def initialize():
    return WAE
