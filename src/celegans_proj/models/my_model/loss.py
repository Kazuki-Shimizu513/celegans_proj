

import torch 
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure  as ms_ssim,
    learned_perceptual_image_patch_similarity as lpips,
)
# from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as lpips

class MyLoss(nn.Module):
    """Loss function """
    def __init__(
        self,
        train_models= ["vae", "diffusion", ], 

        loss_image_weight = 1.0,
        loss_latent_weight = 1.0,
        loss_noise_weight = 1.0,
        loss_kl_weight = 1.0,
    ) -> None:
        super().__init__()
        self.train_models = train_models

        self.loss_image_weight = loss_image_weight
        self.loss_latent_weight = loss_latent_weight
        self.loss_noise_weight = loss_noise_weight
        self.loss_kl_weight = loss_kl_weight

    def forward(
        self,
        imgs, pred_imgs,
        latents=None, pred_latents=None,
        noises=None, pred_noises=None,
        posterior=None,
    ) -> torch.Tensor:
        """

        Args:

        Returns:
            Tensor: loss
        """

        loss = self.loss_image_weight * self.compute_recon_loss(pred_imgs, imgs, perceptual="ssim")

        if "vae" in self.train_models \
            and "diffsion" in self.train_models:
            loss += self.loss_latent_weight * self.compute_recon_loss(pred_latents, latents)
            loss += self.loss_noise_weight * self.compute_recon_loss(pred_noises, noises, )
            loss += self.loss_kl_weight * posterior.kl().mean()# kl loss

        elif "diffsion" in self.train_models:
            loss += self.loss_latent_weight * self.compute_recon_loss(pred_latents, latents)
            loss += self.loss_noise_weight * self.compute_recon_loss(pred_noises, noises, )

        elif "vae" in self.train_models:
            loss += self.loss_kl_weight * posterior.kl().mean()# kl loss

        if torch.isnan(loss):
            loss = torch.tensor(0.0, dtype=torch.float32,device=imgs.device,requires_grad=True)
            print(f"nan in loss")

        return loss

    @staticmethod
    def compute_recon_loss(
        pred, target,
        perceptual = None, 
    ):

        assert pred.shape == target.shape,\
            f"{pred.shape=}\t{target.shape=}"

        loss = F.mse_loss(pred, target)

        if perceptual=="ssim":
            loss += 1 - ms_ssim(pred, target, data_range=1.0)
        elif perceptual=="lpips":
            if pred.shape[1] == 1:
                pred = MyLoss.cvt_channel_gray2RGB(pred)
                target = MyLoss.cvt_channel_gray2RGB(target)

            if pred.min() < 0.0 or target.min() < 0.0:
                pred  = MyLoss.min_max_saling(pred)
                target  = MyLoss.min_max_saling(target)

            loss += 1 - lpips(pred, target, 
                              net_type='squeeze', normalize=True,
                              )

        return loss

    @staticmethod
    def cvt_channel_gray2RGB(x):
        return x.repeat(1, 3, 1, 1)

    @staticmethod
    def min_max_scaling(x, eps=1e-10):
        new_x = (x-x.min())/ (x.max() - x.min() + eps)
        return new_x





if __name__ == "__main__":
    print()
    img = torch.randn(2,1,256,256)
    pred_img = torch.randn(2,1,256,256)

    model = MyTorchModel()
    posterior = model.pipe.vae.encode(img,return_dict=False)[0]
    latents = posterior.sample()
    pred_latents = latents.detach().clone() 

    noise = torch.randn(2,64,32,32)
    pred_noise = torch.randn(2,64,32,32)

    loss_fn = MyLoss(
        train_models= ["vae", "diffusion", ], 
        loss_image_weight = 1.0,
        loss_latent_weight = 1.0,
        loss_noise_weight = 1.0,
        loss_kl_weight = 1.0,
    ) 
    loss = loss_fn(
        imgs, pred_imgs,
        latents, pred_latents,
        noises, pred_noises,
        posterior,
    )
    print(loss)

