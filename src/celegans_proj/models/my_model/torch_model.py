
import sys
import os
import shutil
import math
import random
import pickle
from pathlib import Path
from collections import defaultdict
from statistics import mean  # , stdev  #

import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure  as ms_ssim
import numpy as np


import cv2
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from diffusers import (
    # FlowMatchEulerDiscreteScheduler,
    DDPMScheduler,
    StableDiffusionSAGPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)

# from myProject.utils.diffseg_utils import(
#     vis_without_label,
# )
# from myProject.utils.visualize_results import (  
#     check_imgs, 
#     cvt_RGB2GRAY,
#     strip2numpyImg, 
#     plot_each_image, 
#     plot_2img,
#     plot_all_anomality, 
#     plot_kotai_result,
# )

from .components import (
    cross_attn_init, 
    CrossAttnStoreProcessor,
    register_cross_attention_hook, 
    get_net_attn_map, 
    DiffSeg, 
)







class MyTorchModel(nn.Module):
    def __init__(
            self, 
            train_models=["VAE", "DDPM"],

            # VAE
            in_channels = 1,
            out_channels = 1,
            layers_per_block = 2,
            block_out_channels = (64, 128, 256,256),
            down_block_types = (
                                "DownEncoderBlock2D",
                                "DownEncoderBlock2D",
                                "DownEncoderBlock2D",
                                "DownEncoderBlock2D",
                            ),
            up_block_types = (
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                            ),
            act_fn = "silu",
            latent_channels = 64,
            norm_num_groups = 64,
            sample_size = 256,
            scaling_factor = 0.18215,
            force_upcast = True,

            # DDPM Unet
            down_block_types = (
                                "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D",
                                "CrossAttnDownBlock2D",
                                ),
            mid_block_type = "UNetMidBlock2DCrossAttn",
            up_block_types = (
                                "CrossAttnUpBlock2D", 
                                "CrossAttnUpBlock2D", 
                                "CrossAttnUpBlock2D"
                                "CrossAttnUpBlock2D"
                             ),
            attention_head_dim = (5,10,20,20),
            cross_attention_dim= 1024,
            resnet_time_scale_shift = "default",
            time_embedding_type = "positional",
            conv_in_kernel = 3,
            conv_out_kernel = 3,
            attention_type = "default",

            # DDPM Scheduler
            ddpm_num_steps=1000,
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
            KL_THRESHOLD=[0.0027]*3,
            NUM_POINTS=2,
            REFINEMENT=True,

            # Reproducibility
            seed =44,
 
    ) -> None:
        super().__init__()

        self.train_models = train_models
        self.generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        self.ddpm_num_steps = ddpm_num_steps
        self.guidance_scale=guidance_scale
        self.sag_scale=sag_scale


        vae = AutoencoderKL(
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
                act_fn=act_fn,
                latent_channels=latent_channels,
                norm_num_groups=norm_num_groups,
                sample_size=sample_size,
                scaling_factor=scaling_factor,
                force_upcast=force_upcast,
        )

        unet = UNet2DConditionModel(
                in_channels = latent_channels,
                out_channels = latent_channels,
                layers_per_block=layers_per_block,
                block_out_channels=block_out_channels,
                down_block_types=down_block_types,
                mid_block_type=mid_block_type,
                up_block_types=up_block_types,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                act_fn=act_fn,
                resnet_time_scale_shift=resnet_time_scale_shift,
                time_embedding_type=time_embedding_type,
                attention_type=attention_type,
                conv_in_kernel=conv_in_kernel,
                conv_out_kernel=conv_out_kernel,
        )

        scheduler = DDPMScheduler(
            num_train_timesteps=ddpm_num_steps,
            beta_start =beta_start, 
            beta_end =beta_end,
            trained_betas =trained_betas,
            beta_schedule=beta_schedule,
            variance_type=variance_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type=prediction_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
        )

        sd_sag_pipe = StableDiffusionSAGPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        text_encoder = sd_sag_pipe.text_encoder
        tokenizer = sd_sag_pipe.tokenizer
        # vae = sd_sag_pipe.vae
        # unet = sd_sag_pipe.unet
        # scheduler = sd_sag_pipe.scheduler
        safety_checker = sd_sag_pipe.safety_checker
        feature_extractor = sd_sag_pipe.feature_extractor

        self.pipe = StableDiffusionSAGPipeline(
            vae, 
            text_encoder, 
            tokenizer, 
            unet, 
            scheduler, 
            safety_checker,
            feature_extractor,
            requires_safety_checker = False,
        )
        self.pipe.set_progress_bar_config(disable=True)

        self.segmenter = DiffSeg(KL_THRESHOLD, REFINEMENT, NUM_POINTS)


    def forward(self, img, prompt="."):
        """ forward function for my torch model
            args:
                Bach: which contains image, label, mask,
            Returns:
                Bach: which contains image, mask, label, recon_image, pred_mask,  prediction_score, anomaly_map,
        """

        posterior = self.pipe.vae.encode(img, return_dict=False)[0]
        latents = posterior.sample(generator=self.generator)

        # Reconstruction
        pred_noises, noises, pred_latents, noisy_latents = self.diffusion_step(latents, prompt, )
        if "DDPM" not in self.train_models:
            pred_latents = latents.clone()
        pred_img = self.pipe.vae.decode(pred_latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

        # TODO
        pred_mask = self.segment_with_attn_maps(
                latents, img_dirs, kind_names, img_names, clean_images,# pred_images,
                KL_THRESHOLD = self.args.KL_THRESHOLD, # 0.0001
                REFINEMENT = self.args.REFINEMENT, # True,
                NUM_POINTS = self.args.NUM_POINTS, # 24,
                plot = self.args.plot_mid_variables, # True,
            )



        # TODO
        output = self.generate_anomaly_map(img, pred_img,)

        return output
    
    def segment_with_attn_maps(
        self,
        latents, 
        img_dirs, kind_names, img_names,plot=False, 
    ):

        # TODO eject this 
        attn_maps_with_name = {}
        attn_maps_with_name = self.get_attention_map(
            attn_maps_with_name,
            latents, 
            img_dirs, kind_names, img_names, plot=plot,
        )  
        weight_8_with_name, weight_16_with_name, weight_32_with_name = self.split_attn_maps(
            img_names, 
            attn_maps_with_name
        )


        pred_masks = []
        for i in range(len(latents)):
            # ([1, 2, 1024, 1024])
            pred = self.segmentor.segment( 
                                     weight_32_with_name[img_names[i]].detach().numpy(),
                                     weight_16_with_name[img_names[i]].detach().numpy(),
                                     weight_8_with_name[img_names[i]].detach().numpy(),
                            ) # b x 512 x 512
            pred_masks.append(pred)
        pred_masks = torch.stack(pred_masks, dim=0)
        return pred_masks


    def generate_anomaly_map(self, img, pred_img,):
        # TODO

        return output

    def diffusion_step(
        self,
        latents, 
        prompt,
    ) -> tuple[torch.Tensor]:

        """Latent Diffusion Process
             1. prepare prompt embeds
             2. Sample a random timestep and a noise that we'll add to the images
             3. Predict the noise residual
             4. calc Reverce SDE to produce pred_sample images(x_0) 
                from noisy residual and noisy_images (x_t)
        """

        # 1. prepare prompt embeds
        device = self.pipe._execution_device
        bsz = latents.shape[0]  # batch_size
        do_classifier_free_guidance = self.guidance_scale > 1.0
        do_self_attention_guidance = self.sag_scale > 0.0
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            bsz, # num_images_per_prompt,
            do_classifier_free_guidance,
        )

        # 2. Sample a random timestep and a noise that we'll add to the images
        noises = torch.randn(latents.shape).to(latents.device)
        rand_timesteps = torch.randint(1, self.ddpm_num_steps, (bsz,), 
                                      device=latents.device,).long()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noises, rand_timesteps)

        # 3. Predict the noise residual
        pred_noises = self.pipe.unet(noisy_latents, rand_timesteps, 
                            encoder_hidden_states=prompt_embeds,
                        ).sample

        # 4. calc Reverce SDE to produce pred_sample images(x_0) 
        #    from noisy residual and noisy_images (x_t)
        pred_latents = list()
        for pred_noise, t, noisy_latent in zip(pred_noises, rand_timesteps ,noisy_latents):
            self.pipe.scheduler.set_timesteps(t.detach().item(), device=device)
            pred_latent = self.pipe.scheduler.step(pred_noise, t, noisy_latent).pred_original_sample
            pred_latent = pred_latent[0]
            pred_latents.append(pred_latent)
        pred_latents = torch.stack(pred_latents, dim=0)

        return pred_noises, noises, pred_latents, noisy_latents


    def generate(self,mode="Diffusion"):
        from diffusers.models.vae import DiagonalGaussianDistribution
        z = DiagonalGaussianDistribution.sample(generator=self.generator)
        if mode =="Diffusion":
            _, _, z, _ = self.diffusion_step(z, prompt=".", )

        gen_images = self.pipe.vae.decode(
            z,
            generator=self.generator,
        ).sample

        return  gen_images


