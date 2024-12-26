
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .torch_model import MyTorchModel
from .loss import MyLoss

from logging import getLogger

logger = getLogger(__name__)

class MyModel(AnomalyModule):
    """PL Lightning Module for My model Algorithm.

    Args:

   """

    def __init__(
        self,

        training = True,
        training_mask = False,#True,
        learning_rate = 1e-8,
        train_models=["vae", "diffusion"],
        ddpm_num_steps= 10,# 1000,# 5,#

        # Reproducibility
        seed =44,
        pre_trained: bool = True,
        # Development
        out_path = '/mnt/c/Users/compbio/Desktop/shimizudata/test/',
        # device = "cuda:0",


    ) -> None:
        super().__init__()

        # Reproducibility
        self.seed = seed # 44,
        self.training = training # True,
        self.training_mask = training_mask # True,
        self.learning_rate = learning_rate 
        self.train_models= train_models 
        self.ddpm_num_steps= ddpm_num_steps

        # Development
        self.out_path = out_path
        # self.device = device

        self.pre_trained = pre_trained
        self.model: MyTorchModel

        self.loss = MyLoss(
            train_models= self.train_models, 
            loss_image_weight = 1.0,
            loss_latent_weight = 1.0,
            loss_noise_weight = 1.0,
            loss_kl_weight = 1.0,
        )

    def _setup(self) -> None:

        self.model = MyTorchModel(

            # Reproducibility
            seed =self.seed,
            training = self.training,
            training_mask = self.training_mask,
            train_models=self.train_models,

            # Development
            out_path = self.out_path,
            # device=self.device, 

            # Model General
            layers_per_block = 2,
            act_fn = "silu",
            sample_size = 32,

            # VAE
            vae_in_channels = 1,
            vae_out_channels = 1,
            vae_down_block_types = (
                                "DownEncoderBlock2D",
                                "DownEncoderBlock2D",
                                "DownEncoderBlock2D",
                                "DownEncoderBlock2D",
                            ),
            vae_up_block_types = (
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                            ),
            vae_block_out_channels = (64, 128, 256, 256),
            vae_latent_channels = 64,
            vae_norm_num_groups = 64,
            scaling_factor = 1, # 0.18215,
            force_upcast = True,

            # DDPM Unet
            unet_down_block_types = (
                                "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D",
                                "DownBlock2D",
                                ),
            unet_mid_block_type = "UNetMidBlock2DCrossAttn",
            unet_up_block_types = (
                                "UpBlock2D",
                                "CrossAttnUpBlock2D", 
                                "CrossAttnUpBlock2D", 
                                "CrossAttnUpBlock2D",
                             ),
            unet_attention_head_dim = (5, 10, 20, 20),
            # unet_block_out_channels = (32, 64, 128, 256), # 
            unet_block_out_channels = (320, 640, 1280, 1280),# 
            unet_cross_attention_dim= 1024,
            resnet_time_scale_shift = "default",
            time_embedding_type = "positional",
            conv_in_kernel = 3,
            conv_out_kernel = 3,
            attention_type = "default",

            # DDPM Scheduler
            ddpm_num_steps= self.ddpm_num_steps,
            beta_start = 0.0001, 
            beta_end = 0.02,
            trained_betas = None,
            beta_schedule="linear",
            variance_type = 'fixed_small',
            clip_sample = True,
            clip_sample_range = 1.0,
            prediction_type="epsilon",
            timestep_spacing = 'leading',
            steps_offset = 0,

            # StableDiffusionSAG
            model_id = "stabilityai/stable-diffusion-2",
            guidance_scale=0.0, 
            sag_scale=1.0, 

            # DiffSeg
            ## DiffSegParamModel
            # KL_THRESHOLD=[0.0027]*3,
            NUM_POINTS=2,
            REFINEMENT=True,

        )


    def configure_optimizers(self) :
        """Configure optimizers for decoder and bottleneck.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """

        params = list(self.model.pipe.unet.parameters()) \
            + list(self.model.pipe.vae.parameters())

#         if "vae" in self.train_models  and "diffusion" in self.train_models:
#             params = list(self.model.pipe.unet.parameters()) \
#                 + list(self.model.pipe.vae.parameters())
#         elif "vae" in self.train_models:
#             params = self.model.pipe.vae.parameters()
#         else:
#             params = self.model.pipe.unet.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=(0.95, 0.999),
            # eps=1e-100,
            eps=1e-8,
        )

        # from diffusers.optimization import get_scheduler
        # lr_scheduler = get_scheduler(
        #     "cosine",
        #     optimizer=optimizer,
        #     num_warmup_steps=50,
        # )

        opt_settings = [
            {
                "optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": lr_scheduler,
                #     'monitor': 'val_loss',
                #     'interval': 'epoch',
                # }
            },
        ]

        return tuple(opt_settings)

    def on_train_start(self,):
        if "vae" in self.train_models  and "diffusion" in self.train_models:
            pass
        elif "vae" in self.train_models:
            # Unet (conv_in, downsample_block, mid_block, upsample_block, conv_out)
            for param in self.model.pipe.unet.parameters():
                param.requires_grad = False 
        else:# diffusion
            for param in self.model.pipe.vae.encoder.parameters():
                param.requires_grad = False 
            for param in self.model.pipe.vae.decoder.parameters():
                param.requires_grad = True 


    def training_step(self, batch: dict[str, str | torch.Tensor], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step 

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            batch_idx : Input Batch index
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            loss

        """
        del args, kwargs  # These variables are not used.

        loss = self.loss(self.model(batch))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
            batch_idx : Input Batch index
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """

        output = self.model(batch)
        loss = self.loss(output)

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_maps"]
        batch["pred_scores"] = output["pred_scores"]

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return batch

    def predict_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform a prediction step

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
            batch_idx : Input Batch index
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `predict_epoch_end` for feature concatenation.
        """

        output = self.model(batch)

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_maps"]
        batch["pred_scores"] = output["pred_scores"]

        return batch




    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

if __name__ == "__main__":
    print("hello")
