import sys
import os
import shutil
import psutil
import math
import random
import json
import gc
from pathlib import Path
from collections import defaultdict
from statistics import mean  # , stdev  #

import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import v2
from torchvision.utils import (
    draw_segmentation_masks,
    save_image as tv_save_image,
)


import numpy as np

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk

from tqdm.auto import tqdm

from diffusers import (
    # FlowMatchEulerDiscreteScheduler,
    DDPMScheduler,
    StableDiffusionSAGPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPImageProcessor, 
    CLIPVisionConfig,
    CLIPVisionModelWithProjection
    # CLIPConfig,
    # CLIPModel,
)
from anomalib.data.utils.image import save_image# , show_image

from celegans_proj.utils.file_import import  WDDD2FileNameUtil

from celegans_proj.models.my_model.components import (
    CrossAttnStoreProcessor,
    gaussian_blur_2d,
    min_max_scaling,
    DiffSeg, 
)
from celegans_proj.models.my_model.anomaly_map import AnomalyMapGenerator
from celegans_proj.models.my_model.segmentor_param_net import MyDiffSegParamNet

torch.autograd.set_detect_anomaly(True)

class MyTorchModel(nn.Module):
    def __init__(
            self, 
            # Reproducibility
            seed =44,
            training = True,
            training_mask = False,#True,
            train_models=["vae", "diffsion"],


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
                            ),
            vae_up_block_types = (
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                                "UpDecoderBlock2D",
                            ),
            vae_block_out_channels = (64, 128, 256),
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
            # unet_block_out_channels = (320, 640, 1280, 1280),# 
            unet_block_out_channels = (32, 64, 128, 256), # 
            unet_cross_attention_dim= 1024,
            resnet_time_scale_shift = "default",
            time_embedding_type = "positional",
            conv_in_kernel = 3,
            conv_out_kernel = 3,
            attention_type = "default",

            # DDPM Scheduler
            ddpm_num_steps= 10,# 1000,
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
            map_size = (8,8),
            # map_size = (4,4),


            # DiffSeg
            ## DiffSegParamModel
            KL_THRESHOLD=[0.0027]*4,
            NUM_POINTS=2,
            REFINEMENT=True,

            # Development
            out_path = '/mnt/c/Users/compbio/Desktop/shimizudata/test',
            device="cuda",

    ) -> None:
        super().__init__()

        self.training = training
        self.training_mask = training_mask
        self.train_models = train_models
        self.ddpm_num_steps = ddpm_num_steps
        self.guidance_scale=guidance_scale
        self.sag_scale=sag_scale
        self.map_size = map_size
        self.NUM_POINTS=NUM_POINTS
        self.REFINEMENT=REFINEMENT
        self.device = torch.device(device)

        self.out_path = out_path

        self.anomaly_map_generator = AnomalyMapGenerator()

        self.segmentor = DiffSeg(torch.tensor(KL_THRESHOLD).flatten(), self.NUM_POINTS, self.REFINEMENT, )
 
        self.DiffSegParamModel = MyDiffSegParamNet(
            in_channel = 64, # this depends on VAE latent channel
            KL_len = 4,
        )

        vae = AutoencoderKL(
                in_channels=vae_in_channels,
                out_channels=vae_out_channels,
                layers_per_block=layers_per_block,
                block_out_channels=vae_block_out_channels,
                down_block_types=vae_down_block_types,
                up_block_types=vae_up_block_types,
                act_fn=act_fn,
                latent_channels=vae_latent_channels,
                norm_num_groups=vae_norm_num_groups,
                sample_size=sample_size, 
                scaling_factor=scaling_factor,
                force_upcast=force_upcast,
        )

        unet = UNet2DConditionModel(
                in_channels = vae_latent_channels,
                out_channels = vae_latent_channels,
                layers_per_block=layers_per_block,
                down_block_types=unet_down_block_types,
                mid_block_type=unet_mid_block_type,
                up_block_types=unet_up_block_types,
                block_out_channels=unet_block_out_channels,
                attention_head_dim=unet_attention_head_dim,
                cross_attention_dim=unet_cross_attention_dim,
                sample_size=sample_size,
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

        # tokenizer = CLIPTokenizer()
        text_config = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=1024,# 512,
            intermediate_size=2048,
            projection_dim=1024,# 512,
            num_hidden_layers=1,# 12,
            num_attention_heads=8,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            # This differs from `CLIPTokenizer`'s default and from openai/clip
            # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
            pad_token_id=1,
            bos_token_id=49406,
            eos_token_id=49407,
        )
        text_encoder = CLIPTextModel(text_config)
        # feature_extractor = CLIPImageProcessor(
        #     do_resize= True,
        #     size = None,
        #     do_center_crop = True,
        #     do_rescale = True,
        #     rescale_factor = 1 / 255,
        #     do_normalize = True,
        #     image_mean = None,
        #     image_std = None,
        #     do_convert_rgb = True,
        # )
        vision_config = CLIPVisionConfig(
            hidden_size=1024,# 768,
            intermediate_size=3072,
            projection_dim=1024,# 512,
            num_hidden_layers=1,# 12,
            num_attention_heads=8, # 12,
            num_channels=1, # 3,
            image_size=256,# 224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
        )
        image_encoder = CLIPVisionModelWithProjection(vision_config)

        sd_sag_pipe = StableDiffusionSAGPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        # https://github.com/huggingface/transformers/tree/v4.46.2/src/transformers/models/clip
        tokenizer = sd_sag_pipe.tokenizer #CLIPTokenizer
        # text_encoder = sd_sag_pipe.text_encoder #CLIPTextEncoder
        feature_extractor = sd_sag_pipe.feature_extractor #CLIPImageProcessor
        # vae = sd_sag_pipe.vae
        # unet = sd_sag_pipe.unet
        # scheduler = sd_sag_pipe.scheduler
        safety_checker = sd_sag_pipe.safety_checker

        self.check_settings(
            sd_sag_pipe.vae.config, vae.config, 
            "vae_config","my_vae_config"
        )
        self.check_settings(
            sd_sag_pipe.unet.config, unet.config, 
            "unet_config","my_unet_config"
        )
        self.check_settings(
            sd_sag_pipe.scheduler.config, scheduler.config, 
            "scheduler_config","my_scheduler_config"
        )
        # self.store_settings(text_encoder.config, "text_encoder_config")
        # self.store_settings(tokenizer.config, "tokenizer_config")
        # self.store_settings(feature_extractor.config, "feature_extractor_config")
        # self.store_settings(safety_checker.config, "safety_checker_config")

        self.pipe = StableDiffusionSAGPipeline(
            vae, 
            text_encoder, 
            tokenizer, 
            unet, 
            scheduler, 
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker = False,
        )

        self.pipe.set_progress_bar_config(disable=True)

        self.generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        self.pipe.to(device=self.device) # "cuda"
        self.segmentor.to(device=self.device) # "cuda"
        self.DiffSegParamModel.to(device=self.device) # "cuda"

    def generate(self,mode="Diffusion", prompt = "."):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution()
        z = posterior.sample(generator=self.generator)
        if mode =="Diffusion":
            _, _, z, _, _ = self.diffusion_step(z, prompt=".", )

        gen_images = self.pipe.vae.decode(
            z,
            generator=self.generator,
        ).sample

        # self.pipe = self.pipe.to("cuda")
        # gen_images = np.array(self.pipe(prompt=prompt, sag_scale=self.sag_scale).images)

        return  gen_images

    def cvt_channel_gray2RGB(self,x):
        return x.repeat(1, 3, 1, 1)

    def forward(self, batch, prompt="."):
        """ forward function for my torch model
            args:
                Bach: which contains ['image_path', 'label', 'image', 'mask']
            Returns:
                Bach: which contains image, mask, label, pred_image, pred_mask,  prediction_score, anomaly_map,
        """
        path = batch['image_path']
        imgs = batch['image']
        # imgs = self.cvt_channel_gray2RGB(imgs)

        posterior = self.pipe.vae.encode(imgs, return_dict=False)[0]
        latents = posterior.sample(generator=self.generator)

        # Reconstruction
        if "diffusion" not in self.train_models:
            pred_latents = latents.clone()
            # pred_latents= self.pipe( 
            #     prompt = prompt,
            #     num_inference_steps = 50, # self.ddpm_num_steps,
            #     guidance_scale = self.guidance_scale, #  7.5,
            #     sag_scale = self.sag_scale,#  0.75,
            #     num_images_per_prompt = latents.shape[0], # B
            #     generator = self.generator,
            #     latents = latents,
            #     output_type = "latent",
            #     # output_type = "pt",
            #     return_dict = False,
            # )[0]
        else:
            # guidance_scale = 0.0
            # sag_scale = 0.0
 
            if self.training:
                guidance_scale = 0.0
                sag_scale = 0.0
            else:
                guidance_scale = self.guidance_scale
                sag_scale = self.sag_scale

            (
                pred_noises, noises, 
                pred_latents, noisy_latents, 
                pred_masks, KL_THRESHOLDS,
                gen_latents, gen_masks,  gen_KL_THRESHOLDS
            ) = self.diffusion_step(
                latents, 
                prompt, path,
                guidance_scale = guidance_scale, #  7.5,
                sag_scale = sag_scale,#  0.75,
                num_images_per_prompt = 1,  # SAG for each latent
            )

            seg_masks, otsu_thresholds = self.make_ref_seg_msk(batch['image'], check=False,) # True, ) #(B, 256,256)

            gen_imgs = self.pipe.vae.decode(gen_latents/self.pipe.vae.config.scaling_factor, return_dict=False)[0]

        pred_imgs = self.pipe.vae.decode(pred_latents/self.pipe.vae.config.scaling_factor, return_dict=False)[0]


        # register_outputs in the way of dict , not tuple
        output = dict(
            image=batch['image'],
            image_path=batch['image_path'],
            mask=batch['mask'],
            label=batch['label'],

            posterior=posterior,

            latents=latents,
            pred_latents=pred_latents,

            pred_imgs=pred_imgs,
        )


        if "vae" in self.train_models and "diffusion" in self.train_models:
            output.update(
                noises=noises, 
                pred_noises=pred_noises, 

                gen_imgs=gen_imgs,
                gen_latents=gen_latents,

                KL_THRESHOLDS=KL_THRESHOLDS,
                gen_KL_THRESHOLDS=gen_KL_THRESHOLDS,

                seg_masks=seg_masks,
                pred_masks=pred_masks,
                gen_masks=gen_masks,
              )
        elif "vae" in self.train_models:
            pass
        else:# only "diffusion"
            output.update(
                noises=noises, 
                pred_noises=pred_noises, 

                gen_imgs=gen_imgs,
                gen_latents=gen_latents,

                KL_THRESHOLDS=KL_THRESHOLDS,
                gen_KL_THRESHOLDS=gen_KL_THRESHOLDS,

                seg_masks=seg_masks,
                pred_masks=pred_masks,
                gen_masks=gen_masks,
              )

        anomaly_maps, pred_scores = self.anomaly_map_generator(output,)
        output.update(
            anomaly_maps = anomaly_maps,
            pred_scores = pred_scores
        )
        if not self.training:
            self.store_outputs(output, store=True) #False)# # TODO:: move to MyVisualizer CallBack
        if not self.training: # # inference phase
            output =  {"anomaly_maps": output["anomaly_maps"], "pred_scores": output["pred_scores"]}
        return output

    def make_ref_seg_msk(
        self,
        imgs, 
        check=False, # True,
    ): #(B, 256,256)
        # TODO:: make segmentation mask with kl_thresh 
        #    or something like entropy for reference in generation mask
        root = Path(self.out_path)
        seg_masks = []
        otsu_thresholds = []
        imgs = imgs.numpy(force=True)
        imgs = ((imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-10) * 255.0).astype(np.uint8)
        for img in imgs:
            img = np.squeeze(img)
            if check:
                img_hist, bin_edges = np.histogram(img, bins=256, range=(0,256))
                # save_image(image=img_hist, root=root, filename=f"hist.png")
                plt.plot(img_hist)
                plt.savefig(self.out_path + "/hist.png")
                plt.clf();plt.close();

            entr = entropy(img, disk(3)) # apply entropy filter
            entr = ((entr - entr.min()) / (entr.max() - entr.min() + 1e-10) * 255.0).astype(np.uint8)
            if check:
                _entr = np.tile(entr[:, :, np.newaxis], (1, 1, 3))#  W, H -> W, H, 1*3,
                fig = plt.figure()
                plt.imshow(_entr)
                save_image(image=fig, root=root, filename=f"entropy.png")
                plt.clf();plt.close();
            threshold, mask = cv2.threshold(entr, 0, 1, cv2.THRESH_OTSU)
            if check:
                _mask = (mask * 255.0).astype(np.uint8)
                contours, hierarchy = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_with_contours = cv2.drawContours(img, contours,-1,255,3)
                img_with_contours = np.tile(img_with_contours[:, :, np.newaxis], (1, 1, 3))# W, H -> W, H, 1*3
                fig = plt.figure()
                plt.imshow(img_with_contours)
                save_image(image=fig, root=root, filename=f"contour.png")
                plt.clf();plt.close();
            seg_masks.append(torch.tensor(mask, dtype=torch.long, device=self.device))
            otsu_thresholds.append(torch.tensor(threshold, device=self.device)) # requires_grad = True
        seg_masks = torch.stack(seg_masks, dim=0)
        otsu_thresholds = torch.stack(otsu_thresholds, dim=0)
        return seg_masks, otsu_thresholds

    def diffusion_step(
        self,
        latents, 
        prompt,
        path,
        ip_adapter_image = None,
        ip_adapter_image_embeds = None,
        guidance_scale = 2.0,
        sag_scale = 1.0,
        eta=0.0,
        num_images_per_prompt = 1, # Batch_size
    ) -> tuple[torch.Tensor]:

        """Latent Diffusion Process
             1. prepare prompt embeds
                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
             2. ADD Noise to latents and Sample Gaussian Noise 
             3. Prepare for Denoising Loop 
             4. Execute Denoising Loop
                    # classifier-free guidance produces two chunks of attention map
                    # and we only use unconditional one according to equation (25)
                    # in https://arxiv.org/pdf/2210.00939.pdf
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
        do_classifier_free_guidance = guidance_scale > 1.0
        do_self_attention_guidance = sag_scale > 0.0

        # Prepare Image embeddings
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                do_classifier_free_guidance,
            )

            if do_classifier_free_guidance:
                image_embeds = []
                negative_image_embeds = []
                for tmp_image_embeds in ip_adapter_image_embeds:
                    single_negative_image_embeds, single_image_embeds = tmp_image_embeds.chunk(2)
                    image_embeds.append(single_image_embeds)
                    negative_image_embeds.append(single_negative_image_embeds)
            else:
                image_embeds = ip_adapter_image_embeds

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
        noises = torch.randn(latents.shape).to(latents.device)
        # 2.2 ADD Noise to latents for random timestep
        if self.training:
            bsz = latents.shape[0]  # batch_size
            timesteps = torch.randint(1, self.ddpm_num_steps, (bsz,), 
                                          device=latents.device,).long()
            noisy_latents = self.pipe.scheduler.add_noise(latents, noises, timesteps)
        else:
            timesteps = torch.tensor([self.ddpm_num_steps-1] * len(latents))
            noisy_latents = latents.clone()
        pred_noises = torch.zeros_like(noises)

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

        # 2.4. Prepare extra step kwargs. 
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(self.generator, eta)

        # Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )
        added_uncond_kwargs = (
            {"image_embeds": negative_image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 3 Prepare for Denoising Loop 
        # 3.1 Prepare Attn Processor
        # 3.1.1 Store Original
        original_attn_proc = self.pipe.unet.attn_processors
        self.set_attn_processor(attn_proc_name="CrossAttnStoreProcessor")

        # 4. Execute Denoising Loop
        net_attn_maps = []
        num_total = noisy_latents.shape[0] # timesteps.sum().item()
        with self.pipe.progress_bar(total=num_total) as progress_bar:
            for idx, (latent, timestep) in enumerate(zip(noisy_latents, timesteps)):
                latent = latent.unsqueeze(0)
                timestep = int(timestep.item())
                _attn_maps = {}
                attn_maps = {}
                # 4. Execute Denoising Loop
                if self.training:

                    noise_pred, attn_maps,  = self._diffusion_step(
                        latent,
                        do_classifier_free_guidance,
                        timestep,
                        prompt_embeds,
                        attn_maps,
                    )
                    # Update attn_maps
                    _attn_maps = attn_maps
                    pred_noises[idx] = noise_pred.squeeze(0).clone()
                    # compute the original sample x_t -> x_0
                    latent = self.pipe.scheduler.step(noise_pred, timestep, latent, **extra_step_kwargs).pred_original_sample

                else:
                    for i, t in enumerate(list(reversed(range(timestep)))): # Very Heavy
                        noise_pred, attn_maps, = self._sag_step(
                            i, t, latent, prompt_embeds,
                            do_classifier_free_guidance, do_self_attention_guidance, 
                            guidance_scale, sag_scale,
                            attn_maps,
                            added_uncond_kwargs=added_uncond_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                        )
                        # Update attn_maps
                        _attn_maps = self.update_attn_maps(_attn_maps, attn_maps, timestep)# , process)
                        pred_noises[idx] = pred_noises[idx] + noise_pred.squeeze(0) / timestep
                        # compute the previous noisy sample x_t -> x_t-1
                        latent = self.pipe.scheduler.step(noise_pred, t, latent, **extra_step_kwargs).prev_sample

                pred_latents[idx] = latent.squeeze(0).clone()
                net_attn_maps.append(_attn_maps)

                # Update progress bar
                progress_bar.update()
            progress_bar.close()



        # parts segmentation
        if self.training_mask:
            KL_THRESHOLDS = self.DiffSegParamModel(latents)  
            # KL_THRESHOLDS = torch.tensor([[0.9]*4] * latents.shape[0], device=self.device) 
            pred_masks = self.segment_with_attn_maps(net_attn_maps,KL_THRESHOLDS,path, store=True)
        else:
            KL_THRESHOLDS = torch.tensor([[0.9]*4] * latents.shape[0], device=self.device) 
            pred_masks = torch.randint(5,(latents.shape[0],256,256), device=KL_THRESHOLDS.device)
        del net_attn_maps
        gc.collect()





        # generate latents and gen_attn_maps
        self.set_attn_processor(attn_proc_name="CrossAttnStoreProcessor")
        # timesteps = self.scheduler.timesteps
        timesteps = torch.tensor([self.ddpm_num_steps -1 ] * noises.shape[0])
        gen_latents = noisy_latents.clone()
        gen_attn_maps = []
        num_total = noises.shape[0] # timesteps.sum().item()
        with self.pipe.progress_bar(total=num_total) as progress_bar:
            for idx, (noise, timestep) in enumerate(zip(noises, timesteps)):
                noise = noise.unsqueeze(0)
                timestep = int(timestep.item())
                _attn_maps = {}
                attn_maps = {}
                noise_pred, attn_maps, = self._diffusion_step(
                    noise,
                    do_classifier_free_guidance,
                    timestep,
                    prompt_embeds,
                    attn_maps,
                )
                _attn_maps = attn_maps
                gen_latent = self.pipe.scheduler.step(noise_pred, timestep, noise, **extra_step_kwargs).pred_original_sample

                gen_latents[idx] = gen_latent.squeeze(0).clone()
                gen_attn_maps.append(_attn_maps)
                # Update progress bar
                progress_bar.update()
            progress_bar.close()


        # parts segmentation
        if self.training_mask:
            gen_KL_THRESHOLDS = self.DiffSegParamModel(gen_latents)  
            # gen_KL_THRESHOLDS = torch.tensor([[0.9]*4] * latents.shape[0], device=self.device) 
            gen_masks = self.segment_with_attn_maps(gen_attn_maps,gen_KL_THRESHOLDS,path)
        else:
            # gen_KL_THRESHOLDS = torch.tensor([[0.9]*4] * latents.shape[0], device=self.device) 
            gen_masks = torch.randint(5,(latents.shape[0],256,256), device=KL_THRESHOLDS.device)
        del gen_attn_maps
        gc.collect()


        self.pipe.maybe_free_model_hooks()
        # make sure to set the original attention processors back
        self.pipe.unet.set_attn_processor(original_attn_proc)

        return (
            pred_noises, noises, 
            pred_latents, noisy_latents, 
            pred_masks, KL_THRESHOLDS,
            gen_latents, gen_masks,  gen_KL_THRESHOLDS,
        )

    def set_attn_processor(self,attn_proc_name):
        for name, module in self.pipe.unet.named_modules():
            if not name.split('.')[-1].startswith("attn1"):
                continue
            # to initialize for each Atteniton
            if attn_proc_name == "CrossAttnStoreProcessor":
                attn_proc = CrossAttnStoreProcessor()
            else: 
                attn_proc = AttnProcessor2_0()
            module.processor = attn_proc
 
    def get_attn_map(self,attn_maps):
        # Access Attention Processor
        # map_size = None
        mid_attn_map = None
        for name, module in self.pipe.unet.named_modules():
            if not name.split('.')[-1].startswith("attn1"):
                continue
            if hasattr(module.processor, "attn_map"):
                attn_maps[name] = module.processor.attn_map.clone()#.detach().cpu().numpy()
                del module.processor.attn_map
                # print(f"In get_attn_map:{attn_maps[name].shape=}\t")
                # gc.collect()
            if not name.split('.')[0].startswith("mid_block"):
                continue

            mid_attn_map = attn_maps[name].clone().to(device=self.device) # .copy() # 
            # map_size = module.processor.map_size 
            # print(f"In get_attn_map:{map_size=}\t{type(map_size)=}") # (16,1280) #TODO:WTF
            # del module.processor.map_size
        map_size = self.map_size
        # gc.collect()
        return mid_attn_map, attn_maps, map_size

    def update_attn_maps(self, _attn_maps, attn_maps, timestep):# , process):
        if len(list(_attn_maps.keys()))==0: # first loop
            for name in attn_maps.keys():
                _attn_maps[name] = attn_maps[name]/timestep
        else:
            for name in _attn_maps.keys():
                _attn_maps[name] = _attn_maps[name] + attn_maps[name]/timestep
        del attn_maps
        # gc.collect()
        return _attn_maps

    def register_cross_attention_hook(self, key='attn1',):
        for name, module in self.pipe.unet.named_modules():
            if not name.split('.')[-1].startswith(key):
                continue
            hook = module.register_forward_hook(self.hook_fn(name))

    def hook_fn(self,name):
        def forward_hook(module, input, output):
            # if hasattr(module.processor, "attn_map"):
                # attn_maps[name] = module.processor.attn_map
                # del module.processor.attn_map
            e = name.split('.')
            if 'mid_block' in e and 'attn1' in e :
                pass
                # global map_size
                # module.processor.map_size = output[0].shape[-2:]
        return forward_hook

    def _diffusion_step(
        self,
        latent,
        do_classifier_free_guidance,
        timestep,
        prompt_embeds,
        attn_maps,
    ):

        # 4.1 expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, timestep)

        # 4.2 predict the noise residual
        noise_pred = self.pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        mid_attn_map, attn_maps, map_size = self.get_attn_map(attn_maps)

        return noise_pred, attn_maps,



    def _sag_step(
        self, 
        i, t, latent, prompt_embeds,
        do_classifier_free_guidance, do_self_attention_guidance, 
        guidance_scale, sag_scale,
        attn_maps,
        added_uncond_kwargs=None,
        added_cond_kwargs=None,
    ):

        # 4.1 expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

        # 4.2 predict the noise residual
        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_uncond_kwargs,
        ).sample
        mid_attn_map, attn_maps, map_size = self.get_attn_map(attn_maps)


        # 4.3 perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond \
                + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 4.4 perform self-attention guidance with the stored self-attention map
        if do_self_attention_guidance:
            if do_classifier_free_guidance:
                # DDPM-like prediction of x0
                pred_x0 = self.pipe.pred_x0(latent, noise_pred_uncond, t)

                # get the stored attention maps
                uncond_attn, cond_attn = mid_attn_map.chunk(2)

                # self-attention-based degrading of latents
                degraded_latent, attn_mask, attn_map = self.sag_masking(
                    pred_x0, uncond_attn, map_size,
                    torch.Tensor([t]).to(torch.int32), 
                    self.pipe.pred_epsilon(latent, noise_pred_uncond, t)
                )
                uncond_emb, _ = prompt_embeds.chunk(2)
                # forward and give guidance
                degraded_pred = self.pipe.unet(degraded_latent,t,encoder_hidden_states=uncond_emb,
                                added_cond_kwargs=added_uncond_kwargs,
                                ).sample
                noise_pred = noise_pred + sag_scale * (noise_pred_uncond - degraded_pred)
            else:
                # DDIM-like prediction of x0
                pred_x0 = self.pipe.pred_x0(latent, noise_pred, t) # pred_latents0

                # get the stored attention maps
                cond_attn = mid_attn_map

                # self-attention-based degrading of latents
                degraded_latent, attn_mask, attn_map= self.sag_masking(
                    pred_x0, cond_attn, map_size, 
                    torch.Tensor([t]).to(torch.int32), 
                    self.pipe.pred_epsilon(latent, noise_pred, t)
                )

                # forward and give guidance
                degraded_pred = self.pipe.unet(degraded_latent,t,encoder_hidden_states=prompt_embeds,
                                added_cond_kwargs=added_cond_kwargs,
                                ).sample
                noise_pred = noise_pred + sag_scale * (noise_pred - degraded_pred)

        return noise_pred, attn_maps,

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
        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2) # 1,20,16,16
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0

#         print(f"In sag_masking:{attn_map.shape=}\t{original_latents.shape=}") # [20, 16, 16] [1,64,32,32]
#         print(f"In sag_masking:{map_size=}\tattention_head_dim:{h}") # (16,1280), 20
#         print(f"In sag_masking:{attn_mask=}") # (16),


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


 
    def segment_with_attn_maps(self,net_attn_maps,KL_THRESHOLDS, path, store=False):

        pred_masks = []
        for attn_maps, KL_THRESHOLD, p in zip(net_attn_maps, KL_THRESHOLDS, path):
            self.segmentor.set_KL_THRESHOLD(KL_THRESHOLD.flatten())
           
            weights_dict = self.split_attn_maps(attn_maps,detach=False,)

            if not self.training:
                if store:
                    [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(p)
                    self.store_attn_map(weights_dict, out_name_without_pt=filename)# weight_size depends on VAE layer num

            weights_list = [
                weights_dict["weight_64"],
                weights_dict["weight_32"],
                weights_dict["weight_16"],
                weights_dict["weight_8"],
                # weights_dict["weight_4"],
            ]
            weight_ratio = torch.tensor(self.segmentor.get_weight_rato(weights_list)).to(device=self.device)
            pred = self.segmentor.segment(*weights_list, weight_ratio=weight_ratio,) # b x 512 x 512
            pred = pred.squeeze(0)

            if not self.training:
                if store:
                   self.vis_without_label(pred.numpy(force=True),save=True,
                      name=f"{filename}",
                      num_class=len(pred.unique().tolist())+1,
                   )
 
            pred_masks.append(torch.Tensor(pred))
        pred_masks = torch.stack(pred_masks, dim=0)
        return pred_masks

    def split_attn_maps(self, attn_maps,
        detach=True, instance_or_negative=False,batch_size=2,
        ):
        idx = 0 if instance_or_negative else 1
        weight_4=[];weight_8=[];weight_16=[];weight_32=[];weight_64=[];
        for name, attn_map in attn_maps.items():
            attn_map = attn_map.cpu() if detach else attn_map
            attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
            if len(attn_map.shape) == 4:
                attn_map = attn_map.squeeze()
            size = np.sqrt(attn_map.shape[-2])
            # print(f"{name=}\t{attn_map.shape=}\t{size=}")
            if size == 4:
                weight_4.append(attn_map)
            if size == 8:
                weight_8.append(attn_map)
            if size == 16:
                weight_16.append(attn_map)
            if size == 32:
                weight_32.append(attn_map)
            if size == 64:
                weight_64.append(attn_map)

        weights_dict = dict(
            # weight_4=torch.mean(torch.stack(weight_4,dim=0),dim=0, keepdim=True,),
            weight_8=torch.mean(torch.stack(weight_8,dim=0),dim=0, keepdim=True,),
            weight_16=torch.mean(torch.stack(weight_16,dim=0),dim=0, keepdim=True,),
            weight_32=torch.mean(torch.stack(weight_32,dim=0),dim=0, keepdim=True,),
            weight_64=torch.mean(torch.stack(weight_64,dim=0),dim=0, keepdim=True,),
        )
 
        return weights_dict

    # TODO :: move to MyVisualizer Callback
    def store_settings(self, config_dict, title_without_json):
        p = Path(self.out_path)
        p = p.joinpath("configs")
        p.mkdir(parents=True, exist_ok=True,)
        with open(str(p.joinpath(f"{str(title_without_json)}.json")), "w") as f:
            json.dump(config_dict, f, indent=4, sort_keys=True)

    # TODO :: move to MyVisualizer Callback
    def check_settings(self, ref_cfg, cfg, title1, title2):
        self.store_settings(ref_cfg, title1)
        self.store_settings(cfg, title2)
        not_dict = {}
        for k, v in cfg.items():
            if k not in ref_cfg :
                continue
            if v != ref_cfg[k]:
                not_dict[k] = ref_cfg[k]
        self.store_settings(not_dict, "diffs_"+title1+"_"+title2)


    # TODO :: move to MyVisualizer Callback
    def store_outputs(
        self, 
        outputs,
        latent_kind = ['latents', 'pred_latents', 'noises', 'pred_noises', 'gen_latents',],
        img_kind = ['image','pred_imgs',  'gen_imgs', 'anomaly_maps',],
        msk_kind = ['mask', 'seg_masks','pred_masks', 'gen_masks',],
        store=False,
    ):
        out_path = Path(self.out_path)
        p = out_path.joinpath(f'outputs')
        for k, v in outputs.items():
            if k not in [*img_kind, *msk_kind]:# *latent_kind, ]:
                continue
            p = out_path.joinpath(f'{k}')
            p.mkdir(parents=True, exist_ok=True,)
            for idx, pa in enumerate(outputs['image_path']):
                [base_dir, filename, suffix] = WDDD2FileNameUtil.strip_path(pa)
                # print(f"{k}\n\t{filename=}\n\t{v.shape=}\n")
                _v = v[idx]
                # print(f"\t{_v.shape=}\n")
                if k in img_kind:
                    if _v.ndim == 4:
                        _v=_v.squeeze(0)
                    if _v.max() <= 1.0:
                        _v=min_max_scaling(_v) * 255.0
                    if _v.shape[0] == 1:
                        _v = _v.repeat(3, 1, 1) # 1, W, H,-> 1*3, W, H 
                    if _v.shape[0] == 3:
                        _v = _v.permute(1, 2, 0) # C, W, H -> W, H, C,
                    # print(f"\t{_v.shape=}\n")
                    if store:
                        save_image(image=np.array(_v.numpy(force=True),np.uint8), root=p, filename=f"{filename}.png")
                if k in msk_kind:
                    # continue
                    img = outputs['image'][idx].repeat(3, 1, 1)#  * 255.0
                    if _v.dtype != torch.long:
                        _v = _v.to(torch.long)
                    one_hot = F.one_hot(_v)# [W,H] -> [Class,W,H] 
                    # v = torch.argmax(one_hot, dim=1) # [W,H] <- [Class,W,H]
                    one_hot = one_hot.permute(2, 0, 1).to(torch.bool)
                    seg_mask = draw_segmentation_masks(img, masks=one_hot, alpha=0.6)
                    if store:
                        tv_save_image(seg_mask, str(p.joinpath(f"{filename}.png")))

    # TODO :: move to MyVisualizer Callback
    def vis_without_label(self,M,save=False,name="a",num_class=26):
        out_path = Path(self.out_path)
        p = out_path.joinpath(f'pred_segmentation_maps')
        p.mkdir(parents=True, exist_ok=True,)
        fig = plt.figure(figsize=(20, 20))
        ax = plt.subplot(1, 1, 1)
        ax.imshow(M, cmap='jet',alpha=0.5, vmin=-1, vmax=num_class),
        plt.axis("off")
        if save:
            fig.savefig(str(p.joinpath(f"{name}.png")), format='png',bbox_inches='tight')
            plt.close(fig)

    # TODO :: move to MyVisualizer Callback
    def store_attn_map(self,weights_dict, out_name_without_pt="img_name"):
        out_path = Path(self.out_path)
        p = out_path.joinpath(f'attention_maps')
        p.mkdir(parents=True, exist_ok=True,)
        torch.save(weights_dict,str(p.joinpath(f'{out_name_without_pt}.pt')))

if __name__ == "__main__":
    # __debug__ = True # when you execute without any option 

    # image_path      ['/mnt/e/WDDD2_AD/wildType/train/good/wt_N2_081105_03_T28_Z34.tiff']
    # label   tensor([0], device='cuda:0')
    # image   tensor([[[[-1.3911, -1.3935, -1.3924,  ..., -1.3850, -1.3887, -1.3922],
    #                   [-1.3971, -1.3901, -1.3910,  ..., -1.3846, -1.3820, -1.3861],
    #                   [-1.3799, -1.3889, -1.3901,  ..., -1.3844, -1.3852, -1.3824],
    #                   ...,
    #                   [-1.3875, -1.3845, -1.3844,  ..., -1.4095, -1.4101, -1.4099],
    #                   [-1.3790, -1.3942, -1.3882,  ..., -1.4054, -1.4141, -1.4114],
    #                   [-1.3888, -1.3918, -1.3906,  ..., -1.4129, -1.4155, -1.4158]]]],device='cuda:0')
    # mask    tensor([[[0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  ...,
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0],
    #                  [0, 0, 0,  ..., 0, 0, 0]]], device='cuda:0', dtype=torch.uint8)

    out_path = '/mnt/c/Users/compbio/Desktop/shimizudata/test'
    cuda_0 = torch.device("cuda:0")
    training = False 
    training_mask = True 

    B = 1
    C,W,H = (1,256,256)

    img_path = "/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T60_Z49.tiff"
    img = torch.randn(B,C,W,H,).to(device=cuda_0)
    img = Image.open(img_path)
    transforms = v2.Compose([
                    v2.Grayscale(), # Wide resNet has 3 channels
                    v2.PILToTensor(),
                    v2.Resize(
                        size=(W, H), 
                        interpolation=v2.InterpolationMode.BILINEAR
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    # TODO: YouTransform
                    v2.Normalize(mean=[0.485], std=[0.229]),
                ]) 
    img = transforms(img)
    img = img.to(device=cuda_0)
    img = img.unsqueeze(0)
    img = img.repeat(B, 1, 1, 1)# .permute(1, 2, 0) # C*3, W, H -> W, H, C*3,
    print(f"{img.shape=}")

    msk = torch.zeros(B,W,H,).to(device=cuda_0)
    lbl = torch.tensor([np.int64(0)]* B).to(device=cuda_0)


    batch = dict(
        image_path=[img_path]* B,
        image=img,
        mask=msk,
        label=lbl,
    )

    model = MyTorchModel(
            training = training,
            training_mask = training_mask,
            train_models=["vae", "diffusion"],
            ddpm_num_steps= 4,# 1000,
    ).to(device=cuda_0)
    # model = torch.compile(model)

    # output = model.generate()
    output = model(batch)
    print(f"{output.keys()=}") # dict_keys(['image', 'image_path', 'mask', 'label', 'pred_imgs', 'posterior', 'latents', 'pred_latents', 'noises', 'pred_noises', 'gen_imgs', 'gen_latents', 'KL_THRESHOLDS', 'gen_KL_THRESHOLDS', 'pred_masks', 'gen_masks', 'anomaly_maps', 'pred_scores'])
    
    # for k, v in output.items():
    #     print(f"{k}\n\t{v}\n")

