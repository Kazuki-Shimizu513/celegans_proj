

import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision

from diffusers.utils import deprecate
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0
)
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline


from myProject.utils.diffseg_utils import(
    vis_without_label,
)
from myProject.models.segmentor import(
    DiffSeg, 
)






def attn_call(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
    # print("success here attn call")
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    ####################################################################################################
    # (20,4096,77) or (40,1024,77)
    if hasattr(self, "store_attn_map"):
        self.attn_map = attention_probs
    ####################################################################################################
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    ####################################################################################################
    # if self.store_attn_map:
    if hasattr(self, "store_attn_map"):
        hidden_states, attn_map = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # (2,10,4096,77) or (2,20,1024,77)
        self.attn_map = attn_map
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def cross_attn_init():
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call # attn_call is faster
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call

attn_maps = {}
map_size = None
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

        if 'mid_block' in name.split('.'):
            # nonlocal map_size
            global  map_size
            map_size = output[0].shape[-2:]
    return forward_hook

def register_cross_attention_hook(unet, key='attn1',):
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith(key):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name))
    
    return unet

def register_cross_attention_hook_dm(unet, key='attentions',):
    for name, module in unet.named_modules():
        # if not name.split('.')[-1].startswith(key):
        #     continue

        if not hasattr(module, 'processor'):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name))
    
    return unet


def reshape_attn_map(attn_map):
    attn_map = torch.mean(attn_map,dim=0) # mean by head dim: (20,4096,77) -> (4096,77)
    attn_map = attn_map.permute(1,0) # (4096,77) -> (77,4096)
    latent_size = int(math.sqrt(attn_map.shape[1]))
    latent_shape = (attn_map.shape[0],latent_size,-1)
    attn_map = attn_map.reshape(latent_shape) # (77,4096) -> (77,64,64)

    return attn_map # torch.sum(attn_map,dim=0) = [1,1,...,1]



def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens

import math
def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0) # (10, 32*32, 77) -> (32*32, 77)
    attn_map = attn_map.permute(1,0) # (32*32, 77) -> (77, 32*32)

    # print(attn_map.shape, target_size)
    if target_size[0]*target_size[1] != attn_map.shape[1]:
        temp_res = int(math.sqrt(attn_map.shape[1])) # height and width should be same
        temp_size = (temp_res, temp_res) 
        attn_map = attn_map.view(attn_map.shape[0], *temp_size) # (32*32, 77) -> (77, 32,32)
        attn_map = attn_map.unsqueeze(0) # (77,32,32) -> (1,77,32,32)

        attn_map = F.interpolate(
            attn_map.to(dtype=torch.float32),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze() # (77,64,64)
    else:
        attn_map = attn_map.to(dtype=torch.float32) # (77,64,64)

    attn_map = torch.softmax(attn_map, dim=0)
    attn_map = attn_map.reshape(attn_map.shape[0],-1) # (77,64*64)
    return attn_map

def upscale2(attn_map, target_size):
    attn_map = attn_map.permute(1,0) # (4096, 32*32) -> (32*32, 4096)

    if target_size[0]*target_size[1] != attn_map.shape[1]:
        temp_res = int(math.sqrt(attn_map.shape[1])) # height and width should be same
        temp_size = (temp_res, temp_res) 
        # print(attn_map.shape, temp_size)
        attn_map = attn_map.view(attn_map.shape[0], *temp_size) # (32*32, 77) -> (77, 32,32)
        # print(attn_map.shape)
        attn_map = attn_map.unsqueeze(0) # (77,32,32) -> (1,77,32,32)

        attn_map = F.interpolate(
            attn_map.to(dtype=torch.float32),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze() # (77,64,64)
    else:
        attn_map = attn_map.to(dtype=torch.float32) # (77,64,64)

    attn_map = torch.softmax(attn_map, dim=0)
    attn_map = attn_map.reshape(attn_map.shape[0],-1) # (77,64*64)

    return attn_map


def get_net_attn_map_dm(attn_maps_with_name, img_name, 
                     batch_size=2, 
                     instance_or_negative=False, detach=True,
                     key='attentions',
                     ):
    # target_size = (image_size[0]//4, image_size[1]//4) #(64,64)
    # target_size = (256,256) # explode memory
    target_size = (64,64)

    prev_name = ""
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []
    _attn_maps = {}
    # TODO :: may be this part raise error :attn_maps
    for idxx,  (name, attn_map) in enumerate(attn_maps.items()):

        try:
            if not name.split('.')[-2].startswith(key):
                continue
        except:
            continue # TODO；このバケモン処理どうにかする

        if name==prev_name:
            prev_name = name
            name=name+f"_{idxx}"
        else:
            prev_name = name

        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        _attn_maps[name] = attn_map

        if len(attn_map.shape) == 4:
            attn_map = attn_map.squeeze()

        attn_map = upscale(attn_map, target_size) # (10,32*32,32*32) -> (32*32,64*64)
        attn_map = upscale2(attn_map, target_size) # ADD:: (4096, 64*64) -> (64*64, 64*64)
        attn_map = attn_map.sum(1, keepdim=False) # hw1
        attn_map = attn_map.view(*target_size) 
 
        net_attn_maps.append(attn_map) # (10,32*32,77) -> (64,64)

    net_attn_maps = torch.stack(net_attn_maps,dim=0)
    # net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)
    # net_attn_maps = net_attn_maps.reshape(net_attn_maps.shape[0], 64,64) # (77,64*64) -> (77,64,64)

    attn_maps_with_name[img_name] = _attn_maps


    return net_attn_maps, attn_maps_with_name



def get_net_attn_map(attn_maps_with_name, img_name, 
                     batch_size=2, 
                     instance_or_negative=False, detach=True,
                     key='attn1',
                     ):
    # target_size = (image_size[0]//4, image_size[1]//4) #(64,64)
    # target_size = (256,256) # explode memory
    target_size = (64,64)

    prev_name = ""
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []
    _attn_maps = {}
    # TODO :: may be this part raise error :attn_maps
    for idxx,  (name, attn_map) in enumerate(attn_maps.items()):

        # print(f"{name}\t{attn_map.shape=}")
        if not name.split('.')[-1].startswith(key):
            continue

        if name==prev_name:
            prev_name = name
            name=name+f"_{idxx}"
        else:
            prev_name = name
        # print(f"passed :{name}\t{attn_map.shape=}")

        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        _attn_maps[name] = attn_map

        if len(attn_map.shape) == 4:
            attn_map = attn_map.squeeze()

        attn_map = upscale(attn_map, target_size) # (10,32*32,32*32) -> (32*32,64*64)
        attn_map = upscale2(attn_map, target_size) # ADD:: (4096, 64*64) -> (64*64, 64*64)
        attn_map = attn_map.sum(1, keepdim=False) # hw1
        attn_map = attn_map.view(*target_size) 
 
        net_attn_maps.append(attn_map) # (10,32*32,77) -> (64,64)

    net_attn_maps = torch.stack(net_attn_maps,dim=0)
    # net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)
    # net_attn_maps = net_attn_maps.reshape(net_attn_maps.shape[0], 64,64) # (77,64*64) -> (77,64,64)

    attn_maps_with_name[img_name] = _attn_maps


    return net_attn_maps, attn_maps_with_name


def save_net_attn_map(net_attn_maps, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    total_attn_scores = 0
    # for i, (token, attn_map) in enumerate(zip(tokens, net_attn_maps)):
    for i,  attn_map in enumerate(net_attn_maps):
        attn_map_score = torch.sum(attn_map)
        attn_map = attn_map.cpu().numpy()

        h,w = attn_map.shape
        attn_map_total = h*w
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score
        # token = token.replace('</w>','')
        save_attn_map(
            attn_map,
            f'attn{i+1}_sum{attn_map_score:.2f}',
            f"{dir_name}/attn{i+1}_sum{int(attn_map_score*100)}.png"
        )
    # print(f'total_attn_scores: {total_attn_scores}')


def resize_net_attn_map(net_attn_maps, target_size):
    net_attn_maps = F.interpolate(
        net_attn_maps.to(dtype=torch.float32).unsqueeze(0),
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze() # (77,64,64)
    return net_attn_maps


def save_attn_map(attn_map, title, save_path):
    normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(normalized_attn_map)
    image.save(save_path, format='PNG', compression=0)


def split_attn_maps(img_names, attn_maps_with_name):
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
        # print(f"{len(weight_4)=}\t{len(weight_8)=}\t{len(weight_16)=}\t{len(weight_32)=}")
        # weight_4_with_name[img_name] = torch.stack(weight_4,dim=0)
        weight_8_with_name[img_name] = torch.stack(weight_8,dim=0)
        weight_16_with_name[img_name] = torch.stack(weight_16,dim=0)
        weight_32_with_name[img_name] = torch.stack(weight_32,dim=0)
    return weight_8_with_name, weight_16_with_name, weight_32_with_name,




if __name__=="__main__":


    dir_name = "/mnt/c/Users/compbio/Desktop/shimizudata/test_attn_maps"
    img_path = "/mnt/d/WDDD2/TIF_GRAY/wt_N2_081113_01/wt_N2_081113_01_T75_Z35.tiff"

    prompt = "."

    cross_attn_init()

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,
    ) 

    pipe.unet = register_cross_attention_hook(pipe.unet)
    pipe = pipe.to("cuda")

    clean_image_pil = Image.open(img_path)
    clean_image = torchvision.transforms.functional.to_tensor(clean_image_pil)
    clean_image = torchvision.transforms.functional.resize(clean_image, (256,256))
    # channel 1 -> 3
    clean_image_clone_1 = clean_image.clone()
    clean_image_clone_2 = clean_image.clone()
    clean_image = torch.cat((clean_image, clean_image_clone_1, clean_image_clone_2),
                             dim=0) 

    clean_images = torch.unsqueeze(clean_image, 0)
    clean_images = clean_images.to("cuda")
    clean_images = clean_images.half() # float 64 -> float16 same to pipe params
    print(clean_images.shape)

    posterior = pipe.vae.encode(clean_images, return_dict=False)[0]
    latent = posterior.sample()
    image = pipe(prompt, latents= latent, height=256, width=256, ).images[0]

    attn_maps_with_name={}; img_name='test_name';
    net_attn_maps, attn_maps_with_name = get_net_attn_map(attn_maps_with_name, img_name, batch_size=2,)
    print(net_attn_maps.shape)
    # net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)

    save_net_attn_map(net_attn_maps, dir_name, )
    clean_image_pil.save(dir_name + '/clean.png')
    image.save(dir_name + '/test.png')

    KL_THRESHOLD = [0.05]*3
    REFINEMENT = True
    NUM_POINTS = 24
    img_names = [img_name,]

    weight_8_with_name, weight_16_with_name, weight_32_with_name, =  split_attn_maps(img_names, attn_maps_with_name)
    segmentor = DiffSeg(KL_THRESHOLD, REFINEMENT, NUM_POINTS)
    pred = segmentor.segment( weight_32_with_name[img_names[0]], weight_16_with_name[img_names[0]], weight_8_with_name[img_names[0]]) # b x 512 x 512

    image_np = np.array(clean_image_pil)
    print(image_np.shape, pred.shape)

