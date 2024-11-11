
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
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure  as ms_ssim,
)
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
    CrossAttnStoreProcessor,
    gaussian_blur_2d,
    min_max_scaling,
    cross_attn_init, 
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

        cross_attn_init()
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

    def generate(self,mode="Diffusion"):
        from diffusers.models.vae import DiagonalGaussianDistribution
        z = DiagonalGaussianDistribution.sample(generator=self.generator)
        if mode =="Diffusion":
            _, _, z, _, _ = self.diffusion_step(z, prompt=".", )

        gen_images = self.pipe.vae.decode(
            z,
            generator=self.generator,
        ).sample

        return  gen_images

    def forward(self, img, prompt="."):
        """ forward function for my torch model
            args:
                Bach: which contains image, label, mask,
            Returns:
                Bach: which contains image, mask, label, pred_image, pred_mask,  prediction_score, anomaly_map,
        """

        posterior = self.pipe.vae.encode(img, return_dict=False)[0]
        latents = posterior.sample(generator=self.generator)

        # Reconstruction
        pred_noises, noises, pred_latents, noisy_latents , attn_maps= self.diffusion_step(latents, prompt, )
        if "DDPM" not in self.train_models:
            pred_latents = latents.clone()
        pred_img = self.pipe.vae.decode(pred_latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

        # TODO :: eject domain names
        pred_mask = self.segment_with_attn_maps(
                latents, 
                img_dirs, kind_names, img_names
        )

        # TODO
        output = self.generate_anomaly_map(img, pred_img, pred_mask)

        return output

    def diffusion_step(
        self,
        latents, 
        prompt,
        eta=0.0,
    ) -> tuple[torch.Tensor]:

        """Latent Diffusion Process
             1. prepare prompt embeds
                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
             2. ADD Noise to latents and Sample Gaussian Noise 
             3. Prepare for Denoising Loop 
             4. Execute Denoising Loop
             5. calc Reverce SDE to produce pred_sample images(x_0) 
                from noisy residual and noisy_images (x_t)
        """

        # 1. prepare prompt embeds
        # 1.1 Default height and width to unet
        height =  self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width =  self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor

        # 1.2 Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self.pipe._execution_device
        do_classifier_free_guidance = self.guidance_scale > 1.0
        do_self_attention_guidance = self.sag_scale > 0.0
        bsz = latents.shape[0]
        num_images_per_prompt = bsz

        # 1.3 Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])



        # 2. ADD Noise to latents and Sample Gaussian Noise 
        # 2.1 Sample a random timestep and Nose
        self.pipe.scheduler.set_timesteps(self.ddpm_num_steps, device=device)
        bsz = latents.shape[0]  # batch_size
        timesteps = torch.randint(1, self.ddpm_num_steps, (bsz,), 
                                      device=latents.device,).long()
        noises = torch.randn(latents.shape).to(latents.device)
        # 2.2 ADD Noise to latents for random timestep
        noisy_latents = self.pipe.scheduler.add_noise(latents, noises, timesteps)

        # 2.3 Prepare latent variables with prompt_embeds
        num_channels_latents = self.pipe.unet.config.in_channels
        noisy_latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            self.generator,
            noisy_latents,
        )
        pred_latents = noisy_latents.clone()
        pred_noises = torch.zeros_like(noises)

        # 2.4. Prepare extra step kwargs. 
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(self.generator, eta)

        # 3 Prepare for Denoising Loop 
        # 3.1 Prepare Attn Processor
        # TODO:: Store All Attention Map for DiffSeg and Sag
        # 3.1.1 Store Original
        original_attn_proc = self.pipe.unet.attn_processors

        # 3.1.2 Replace my Attn Processor
        cross_attn_init()
        # store_processor = CrossAttnStoreProcessor()
        # self.pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor
        # store_processor = self.pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor
        # # self.pipe.maybe_free_model_hooks()
        # # self.pipe.unet.set_attn_processor(store_processor)

        # 3.1.3 Register hook function 
        attn_maps = {}
        map_size = None
        self.pipe.unet = register_cross_attention_hook(self.pipe.unet)

        # 4. Execute Denoising Loop
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for idx, (latent, timestep) in enumerate(zip(noisy_latents, timesteps)):
                for i, t in enumerate(list(reversed(range(timestep)))):
                    # 4.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latent] * 2) \
                        if do_classifier_free_guidance else latent
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                    # 4.2 predict the noise residual
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                    ).sample

                    # 4.3 perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # 4.4 perform self-attention guidance with the stored self-attention map
                    if do_self_attention_guidance:
                        # classifier-free guidance produces two chunks of attention map
                        # and we only use unconditional one according to equation (25)
                        # in https://arxiv.org/pdf/2210.00939.pdf
                        if do_classifier_free_guidance:
                            # DDPM-like prediction of x0
                            pred_x0 = self.pipe.pred_x0(latents, noise_pred_uncond, t)
                            # get the stored attention maps
                            # uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)
                            uncond_attn, cond_attn = store_processor.attn_map.chunk(2)
                            # self-attention-based degrading of latents
                            # degraded_latents = self.pipe.sag_masking(
                            degraded_latents, attn_mask, attn_map = self.sag_masking(
                                pred_x0, uncond_attn, map_size, t, self.pipe.pred_epsilon(latent, noise_pred_uncond, t)
                            )
                            uncond_emb, _ = prompt_embeds.chunk(2)
                            # forward and give guidance
                            degraded_pred = self.pipe.unet(
                                degraded_latents,
                                t,
                                encoder_hidden_states=uncond_emb,
                            ).sample
                            noise_pred += self.sag_scale * (noise_pred_uncond - degraded_pred)
                        else:
                            # DDIM-like prediction of x0
                            pred_x0 = self.pipe.pred_x0(latent, noise_pred, t) # latents_x0
                            # get the stored attention maps
                            # cond_attn = store_processor.attention_probs
                            cond_attn = store_processor.attn_map
                            # self-attention-based degrading of latents
                            # degraded_latents = self.pipe.sag_masking(
                            degraded_latents, attn_mask, attn_map= self.sag_masking(
                                pred_x0, cond_attn, map_size, t, self.pipe.pred_epsilon(latent, noise_pred, t)
                            )
                            # forward and give guidance
                            degraded_pred = self.pipe.unet(
                                degraded_latents,
                                t,
                                encoder_hidden_states=prompt_embeds,
                            ).sample
                            noise_pred += self.sag_scale * (noise_pred - degraded_pred)
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    latent = self.pipe.scheduler.step(noise_pred, t, latent, **extra_step_kwargs).prev_sample
                    pred_noises[idx] += noise_pred
                pred_latents[idx] = latent

        self.pipe.maybe_free_model_hooks()
        # make sure to set the original attention processors back
        self.pipe.unet.set_attn_processor(original_attn_proc)


        return pred_noises, noises, pred_latents, noisy_latents, attn_maps



    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        target_size = (64,64)
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.pipe.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]
        # Add #############################################################
        if isinstance(h, tuple):
            h = h[-1]
        ##################################################################

        # self.logger.info(f"{attn_map.shape=}\t{original_latents.shape=}\tattention_head_dim:{h}") # [80, 16, 16] [4,1024,32,32], 20
        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2) # 4,20,16,16
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0

        # Add #############################################################
        _attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            # .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        _attn_mask = F.interpolate(_attn_mask, target_size,
                                    mode='bilinear',
                                    align_corners=False,
                                   )
        ##################################################################

        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Add #############################################################
        _attn_map = attn_map.mean(1, keepdim=False) # h
        _attn_map = _attn_map.sum(1, keepdim=False) # hw1
        _attn_map = (
            _attn_map.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            # .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        _attn_map = F.interpolate(_attn_map, target_size,
                                    mode='bilinear',
                                    align_corners=False,
                                   )
 
        ##################################################################


        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # Noise it again to match the noise level
        degraded_latents = self.pipe.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t[None])

        return degraded_latents, _attn_mask, _attn_map


 
    def segment_with_attn_maps(
        self,
        latents, 
        img_dirs, kind_names, img_names,
    ):

        # TODO eject this 
        attn_maps_with_name = self.get_attention_map(
            latents, 
            img_dirs, kind_names, img_names,
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

    def get_attention_map(
            self, 
            latents,
            img_dirs, kind_names, img_names 
    ):

        # TODO :: eject this inference code to my guide diffusion call
        # attn_maps = {}
        # cross_attn_init()
        # self.pipe.unet = register_cross_attention_hook(self.pipe.unet)

        # images = self.pipe(
        #         self.args.prompt, 
        #         latents = latents,
        #         width=256,height=256,
        #         num_images_per_prompt = self.args.batch_size,
        #         num_inference_steps=self.args.ddpm_num_inference_steps,
        #         guidance_scale= self.args.guidance_scale,
        #         sag_scale= 0, # self.args.sag_scale,
        #         generator=self.generator,
        #         output_type="np",
        # ).images

        # TODO :: simplify this code
        attn_maps_with_name = {}
        for img_dir, kind_name, img_name  in zip(
            img_dirs, kind_names, img_names 
        ):
            net_attn_maps, attn_maps_with_name = get_net_attn_map(attn_maps_with_name, img_name, batch_size=2,)
        return attn_maps_with_name

    def split_attn_maps(self, img_names, attn_maps_with_name):
        detach=True; instance_or_negative=False;batch_size=2;
        idx = 0 if instance_or_negative else 1
        weight_4_with_name={};weight_8_with_name={};weight_16_with_name={};weight_32_with_name={};
        for img_name in img_names:
            # print(f"attn_maps: {len(attn_maps_with_name[img_name])=}")
            weight_4=[];weight_8=[];weight_16=[];weight_32=[];
            for name, attn_map in attn_maps_with_name[img_name].items():

                attn_map = attn_map.cpu() if detach else attn_map
                attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
                if len(attn_map.shape) == 4:
                    attn_map = attn_map.squeeze()
                size = np.sqrt(attn_map.shape[-2])
                # print(f"{name=}\t{attn_map.shape=}\t{size=}")
                # if size == 4:
                #     weight_4.append(attn_map)
                if size == 8:
                    weight_8.append(attn_map)
                if size == 16:
                    weight_16.append(attn_map)
                if size == 32:
                    weight_32.append(attn_map)
            # print(f"{len(weight_8)=}\t{len(weight_16)=}\t{len(weight_32)=}")
            # weight_8_with_name[img_name] = torch.stack(weight_8,dim=0)
            weight_8 = torch.mean(torch.stack(weight_8,dim=0),dim=0, keepdim=True,) # (5,2,64,64) -> (1,2,64,64)
            weight_16 = torch.mean(torch.stack(weight_16,dim=0),dim=0, keepdim=True,)
            weight_32 = torch.mean(torch.stack(weight_32,dim=0),dim=0, keepdim=True,)

            # print(f"{weight_8.shape=}\t{weight_16.shape=}\t{weight_32.shape=}")
            # print(f"{weight_8.min()=}\t{weight_8.max()=}")
            weight_8_with_name[img_name] = min_max_scaling(weight_8)
            weight_16_with_name[img_name] = min_max_scaling(weight_16)
            weight_32_with_name[img_name] = min_max_scaling(weight_32)

        return weight_8_with_name, weight_16_with_name, weight_32_with_name,


    def generate_anomaly_map(self, img, pred_img,):
        # TODO

        return output


