from __future__ import annotations

import os
import math
import random
import logging
import inspect
import argparse
import datetime

from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torchvision
import torch.nn.functional as F

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from enfugue.diffusion.train.motion.director.utils.util import save_videos_grid, load_diffusers_lora, load_weights
from enfugue.diffusion.train.motion.director.utils.lora_handler import LoraHandler
from enfugue.diffusion.train.motion.director.utils.lora import extract_lora_child_module
from enfugue.diffusion.train.motion.director.utils.dataset import VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset, CachedDataset, VID_TYPES
from enfugue.diffusion.train.motion.director.utils.configs import get_simple_config
from lion_pytorch import Lion

augment_text_list = [
    "a video of",
    "a high quality video of",
    "a good video of",
    "a nice video of",
    "a great video of",
    "a video showing",
    "video of",
    "video clip of",
    "great video of",
    "cool video of",
    "best video of",
    "streamed video of",
    "excellent video of",
    "new video of",
    "new video clip of",
    "high quality video of",
    "a video showing of",
    "a clear video showing",
    "video clip showing",
    "a clear video showing",
    "a nice video showing",
    "a good video showing",
    "video, high quality,"
    "high quality, video, video clip,",
    "nice video, clear quality,",
    "clear quality video of"
]

def create_save_paths(output_dir: str):
    lora_path = f"{output_dir}/lora"

    directories = [
        output_dir,
        f"{output_dir}/samples",
        f"{output_dir}/sanity_check",
        lora_path
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    return lora_path

def get_train_dataset(dataset_types, train_data, tokenizer):
    def process_folder_of_videos(train_datasets: list, video_folder: str):
         for video_file in os.listdir(video_folder):

            is_video = any([video_file.split(".")[-1] in ext for ext in VID_TYPES])

            if is_video:
                train_data["single_video_path"] = f"{video_folder}/{video_file}"
                train_datasets.append(SingleVideoDataset(**train_data, tokenizer=tokenizer))

    train_datasets = []

    # Loop through all available datasets, get the name, then add to list of data to process.
    for DataSet in [VideoJsonDataset, SingleVideoDataset, ImageDataset, VideoFolderDataset]:
        for dataset in dataset_types:
            if dataset == DataSet.__getname__():
                video_folder = train_data.get("path", "")

                if os.path.exists(video_folder) and dataset == "folder":
                            process_folder_of_videos(
                                train_datasets, 
                                video_folder
                            )
                            continue
                train_datasets.append(DataSet(**train_data, tokenizer=tokenizer))

    if len(train_datasets) > 0:
        return train_datasets
    else:
        raise ValueError("Dataset type not found: 'json', 'single_video', 'folder', 'image'")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def get_cached_latent_dir(c_dir):
    from omegaconf import ListConfig

    if isinstance(c_dir, str):
        return os.path.abspath(c_dir) if c_dir is not None else None
    
    if isinstance(c_dir, ListConfig):
        c_dir = OmegaConf.to_object(c_dir)
        return c_dir

    return None

def handle_cache_latents(
        should_cache, 
        output_dir, 
        train_dataloader, 
        train_batch_size, 
        vae, 
        cached_latent_dir=None,
        shuffle=False,
        minimum_required_frames=16,
        sampler=None,
        device='cuda'
    ):

    # Cache latents by storing them in VRAM. 
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: 
        return None
    
    vae_dtype = vae.dtype
    vae.to(device, dtype=torch.float32)

    if hasattr(vae, 'enable_slicing'):
        vae.enable_slicing()
    
    cached_latent_dir = get_cached_latent_dir(cached_latent_dir)

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)
    
        for i, batch in enumerate(train_dataloader):

            frames = batch['pixel_values'].shape[1]

            not_min_frames = frames > 2 and frames < minimum_required_frames
            not_img_train = (frames == 1 and batch['dataset'] != 'image')

            if any([not_min_frames, not_img_train]) and minimum_required_frames != 0:
                print(f"""
                    Batch item at index {i} does not meet required minimum frames.
                    Seeing this error means that some of your video lengths are too short, but training will continue.
                    Minimum Frames: {minimum_required_frames}
                    Batch item frames: Batch index = {i}, Batch Frames = {frames}
                    """
                )
                continue

            save_name = f"cached_{i}"
            full_out_path =  f"{cache_save_dir}/{save_name}.pt"
            pixel_values = batch['pixel_values'].to(device, dtype=torch.float32)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            
            for k, v in batch.items(): 
                batch[k] = v[0]

            torch.save(batch, full_out_path)
            
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir
        
    # Convert string to list of strings for processing if we have more than.
    cache_save_dir = (
                [cache_save_dir] if not isinstance(cache_save_dir, list) 
            else 
                cache_save_dir
        )

    cached_dataset_list = []

    for save_dir in cache_save_dir:
        cached_dataset = CachedDataset(cache_dir=save_dir)
        cached_dataset_list.append(cached_dataset)

    if len(cached_dataset_list) > 1:
        print(f"Found {len(cached_dataset_list)} cached datasets. Merging...")
        new_cached_dataset = torch.utils.data.ConcatDataset(cached_dataset_list)
    else:
        new_cached_dataset = cached_dataset_list[0] 

    vae.to(dtype=vae_dtype)

    return torch.utils.data.DataLoader(
                new_cached_dataset,
                batch_size=train_batch_size, 
                shuffle=shuffle,
                num_workers=2,
                persistent_workers=True,
                pin_memory=False,
                sampler=sampler
            )

def do_sanity_check(
    batch: Dict, 
    cache_latents: bool, 
    validation_pipeline: AnimationPipeline, 
    device: str, 
    image_finetune: bool=False,
    output_dir: str = "",
    dataset_id: int = 0
):
    pixel_values, texts = batch['pixel_values'].cpu(), batch["text_prompt"]
    
    if cache_latents:
        pixel_values = validation_pipeline.decode_latents(batch["pixel_values"].to(device))
        to_torch = torch.from_numpy(pixel_values)
        pixel_values = rearrange(to_torch, 'b c f h w -> b f c h w')
        
    if not image_finetune:
        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
            pixel_value = pixel_value[None, ...]
            text = f"{str(dataset_id)}_{text}"
            save_name = f"{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'-{idx}'}.mp4"
            save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{save_name}", rescale=not cache_latents)
    else:
        for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
            pixel_value = pixel_value / 2. + 0.5
            text = f"{str(dataset_id)}_{text}"
            save_name = f"{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'-{idx}'}.png"
            torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{save_name}")

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue
            
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params

def scale_loras(lora_list: list, scale: float, step=None, spatial_lora_num=None):
    
    # Assumed enumerator
    if step is not None and spatial_lora_num is not None:
        process_list = range(0, len(lora_list), spatial_lora_num)
    else:
        process_list = lora_list

    for lora_i in process_list:
        if step is not None:
            lora_list[lora_i].scale = scale
        else:
            lora_i.scale = scale

def get_spatial_latents(
        batch: Dict, 
        random_hflip_img: int, 
        cache_latents: bool,
        noisy_latents:torch.Tensor, 
        target: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler: DDPMScheduler
    ):
    ran_idx = torch.randint(0, batch["pixel_values"].shape[2], (1,)).item()
    use_hflip = random.uniform(0, 1) < random_hflip_img

    noisy_latents_input = None
    target_spatial = None

    if use_hflip:
        pixel_values_spatial = torchvision.transforms.functional.hflip(
            batch["pixel_values"][:, ran_idx, :, :, :] if not cache_latents else\
                batch["pixel_values"][:, :, ran_idx, :, :]
        ).unsqueeze(1)

        latents_spatial = (
            tensor_to_vae_latent(pixel_values_spatial, vae) if not cache_latents
            else
            pixel_values_spatial
        )

        noise_spatial = sample_noise(latents_spatial, 0,  use_offset_noise=use_offset_noise)
        noisy_latents_input = noise_scheduler.add_noise(latents_spatial, noise_spatial, timesteps)

        target_spatial = noise_spatial
    else:
        noisy_latents_input = noisy_latents[:, :, ran_idx, :, :]
        target_spatial = target[:, :, ran_idx, :, :]

    return noisy_latents_input, target_spatial, use_hflip

def create_ad_temporal_loss(
        model_pred: torch.Tensor, 
        loss_temporal: torch.Tensor, 
        target: torch.Tensor
    ):
    beta = 1
    alpha = (beta ** 2 + 1) ** 0.5

    ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()

    model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
    target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2)

    loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
    loss_temporal = loss_temporal + loss_ad_temporal

    return loss_temporal
