from __future__ import annotations
# Adapted from https://github.com/ExponentialML/AnimateDiff-MotionDirector/blob/main/train.py
import os
import math
import random

from collections import OrderedDict
from dataclasses import dataclass, field
from copy import deepcopy

from enfugue.util import logger, get_step_callback

from typing import Optional, Literal, Dict, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import device as TorchDevice

@dataclass
class MotionTrainer:
    # Model paths
    checkpoint_path: str
    motion_module_path: str
    # Other models
    lora_path: Optional[Union[str, Tuple[str, float], List[Union[str, Tuple[str, float]]]]]=None
    domain_adapter_path: Optional[str]=None
    # Training
    batch_size: int=1
    max_training_epochs: int=503
    target_spatial_modules: List[str] = field(default_factory=list)
    target_temporal_modules: List[str] = field(default_factory=list)
    # LoRA arguments
    use_motion_lora_format: bool=True
    lora_rank: int=32
    lora_unet_dropout: float=0.1
    # Learning arguments
    learning_rate: float=0.00005
    learning_rate_spatial: float=0.00001
    adam_weight_decay: float=0.1
    adam_beta1: float=0.9
    adam_beta2: float=0.999
    max_grad_norm: float=1.0
    use_lion_optimization: bool=True
    use_offset_noise: bool=False
    mixed_precision_training: bool=True
    use_bucketing: bool=True
    fps: int=0
    frame_step: int=1
    width: int=256
    height: int=256
    lr_scheduler: str="constant"
    random_null_text_ratio: float=0.0
    max_chunks: int=0
    random_hflip_img: int=-1
    # Checkpointing arguments
    checkpointing_steps: int=100
    seed: int=42
    # Validation arguments
    validation_spatial_scale: float=0.5
    validation_steps: int=50
    sample_steps: int=25
    sample_guidance_scale: float=8.5
    sample_width: int=384
    sample_height: int=384
    sample_frames: int=16
    # Configuration dictionaries
    noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.noise_scheduler_kwargs = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "clip_sample": False
        }
        self.target_spatial_modules = ["Transformer3DModel"]
        self.target_temporal_modules = ["TemporalTransformerBlock"]

    @classmethod
    def simple(
        cls,
        quality: Literal["low", "preferred", "best"]="preferred",
        **kwargs: Any
    ) -> MotionTrainer:
        """
        Uses a simple quality preset
        """
        if quality == "low":
            quality_config = {
                "width": 256,
                "height": 256,
                "sample_width": 512,
                "sample_height": 512,
                "use_bucketing": False,
                "lora_rank": 32,
            }
        elif quality == "preferred":
            quality_config = {
                "width": 384,
                "height": 384,
                "sample_width": 384,
                "sample_height": 384,
                "use_bucketing": True,
                "lora_rank": 64,
            }
        elif quality == "best":
            quality_config = {
                "width": 512,
                "height": 512,
                "sample_width": 512,
                "sample_height": 512,
                "use_bucketing": True,
                "lora_rank": 64,
            }
        else:
            raise ValueError(f"Unknown quality preset '{quality}'") # type: ignore[unreachable]

        return cls(**{
            **quality_config,
            **kwargs
        })

    def __call__(
        self,
        name: str,
        cache_dir: str,
        output_dir: str,
        video_path: str,
        training_prompt: str,
        device: Union[str, TorchDevice],
        validation_prompt: Optional[Union[str, List[str]]]=None,
        validation_negative_prompt: Optional[Union[str, List[str]]]=None,
        train_temporal: bool=True,
        train_spatial: bool=True,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
    ) -> List[List[Tuple[int, str, Image]]]:
        """
        Executes the training.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from lion_pytorch import Lion
        from torch.utils.data import DataLoader, ConcatDataset
        from diffusers.optimization import get_scheduler
        from diffusers.schedulers import DDPMScheduler, DDIMScheduler
        from enfugue.diffusion.util import load_state_dict, Video
        from enfugue.diffusion.animate.pipeline import EnfugueAnimateStableDiffusionPipeline
        from enfugue.diffusion.train.motion.director.utils.lora_handler import LoraHandler
        from enfugue.diffusion.train.motion.director.utils.lora import extract_lora_child_module
        from enfugue.diffusion.train.motion.director.utils.convert_lora_safetensor_to_diffusers import (
            convert_lora,
            load_diffusers_lora
        )
        from enfugue.diffusion.train.motion.director import (
            scale_loras,
            sample_noise,
            get_train_dataset,
            create_optimizer_params,
            tensor_to_vae_latent,
            get_spatial_latents,
            create_ad_temporal_loss,
        )

        # Handle folders
        lora_dir = os.path.join(output_dir, "lora")
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(lora_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

        # Instantiate pipeline
        pipeline = EnfugueAnimateStableDiffusionPipeline.from_ckpt(
            checkpoint_path=self.checkpoint_path,
            motion_module=self.motion_module_path,
            cache_dir=cache_dir,
            load_safety_checker=False,
            use_lora_compatible_layers=False, # We'll inject our own
        )

        # Set scheduler
        pipeline.scheduler = DDIMScheduler(**{
            **self.noise_scheduler_kwargs,
            **{"steps_offset": 1}
        })

        # Load domain adapter if specified
        if self.domain_adapter_path:
            logger.debug(f"Loading domain adapter from {self.domain_adapter_path}")
            load_diffusers_lora(pipeline, load_state_dict(self.domain_adapter_path))

        # Load other LoRA if specified
        if self.lora_path:
            if isinstance(self.lora_path, list):
                lora_paths = self.lora_path
            else:
                lora_paths = [self.lora_path]
            for lora_path in lora_paths:
                if isinstance(lora_path, tuple):
                    lora_path, lora_weight = lora_path
                else:
                    lora_weight = 1.0
                convert_lora(pipeline, load_state_dict(lora_path), alpha=lora_weight)

        # Get the pipeline components
        unet = pipeline.unet
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder
        tokenizer = pipeline.tokenizer

        # Make schedulers
        temporal_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)
        spatial_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs)

        # Freeze them for training
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # Enable checkpointing
        unet.enable_gradient_checkpointing()

        # Move to GPU
        vae.to(device)
        text_encoder.to(device)
        unet.to(device)
        pipeline.to(device)

        # Create training data dictionary
        train_data = {
            "single_video_path": video_path,
            "single_video_prompt": training_prompt,
            "max_chunks": self.max_chunks,
            "frame_step": self.frame_step,
            "n_sample_frames": self.sample_frames,
            "width": self.width,
            "height": self.height,
            "sample_width": self.sample_width,
            "sample_height": self.sample_height,
            "sample_size": (self.sample_height, self.sample_width),
            "sample_start_idx": 0,
            "fps": self.fps,
            "use_bucketing": self.use_bucketing,
            "lora_rank": self.lora_rank
        }

        logger.info(f"Creating training dataset using configuration {train_data}")

        # Create dataset
        train_dataset = get_train_dataset(
            dataset_types=["single_video"],
            train_data={
                "single_video_path": video_path,
                "single_video_prompt": training_prompt,
                "max_chunks": self.max_chunks,
                "frame_step": self.frame_step,
                "n_sample_frames": self.sample_frames,
                "width": self.width,
                "height": self.height,
                "sample_width": self.sample_width,
                "sample_height": self.sample_height,
                "sample_size": (self.sample_height, self.sample_width),
                "sample_start_idx": 0,
                "fps": self.fps,
                "use_bucketing": self.use_bucketing,
                "lora_rank": self.lora_rank
            },
            tokenizer=tokenizer,
        )

        if len(train_dataset) > 0:
            train_dataset = ConcatDataset(train_dataset)
        else:
            train_dataset = train_dataset[0]

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True
        )

        if len(train_dataloader) < 1:
            raise ValueError("No data to load.")

        # Calculate steps
        max_training_steps = self.max_training_epochs * len(train_dataloader)

        # Temporal LoRA
        lora_manager_temporal = LoraHandler(
            use_unet_lora=True,
            unet_replace_modules=self.target_temporal_modules
        )
        unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
            use_lora=True,
            model=unet,
            replace_modules=lora_manager_temporal.unet_replace_modules,
            dropout=0.0,
            lora_path=os.path.join(lora_dir, "temporal"),
            r=self.lora_rank
        )

        # Temporal Optimizer
        optimizer_temporal = Lion(
            create_optimizer_params([OrderedDict([
                ("model", unet_lora_params_temporal),
                ("condition", True),
                ("extra_params",  {"lr": self.learning_rate}),
                ("is_lora", True),
                ("negation", None),
            ])], self.learning_rate),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay
        )

        # Temporal scheduler
        lr_scheduler_temporal = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer_temporal,
            num_warmup_steps=0,
            num_training_steps=max_training_steps
        )

        # Spatial LoRA
        lora_manager_spatial = LoraHandler(
            use_unet_lora=True,
            unet_replace_modules=self.target_spatial_modules
        )
        unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
            use_lora=True,
            model=unet,
            replace_modules=lora_manager_spatial.unet_replace_modules,
            dropout=self.lora_unet_dropout,
            lora_path=os.path.join(lora_dir, "spatial"),
            r=self.lora_rank
        )

        # Spatial Optimizer
        optimizer_spatial = Lion(
            create_optimizer_params([OrderedDict([
                ("model", unet_lora_params_spatial),
                ("condition", True),
                ("extra_params",  {"lr": self.learning_rate_spatial}),
                ("is_lora", True),
                ("negation", None),
            ])], self.learning_rate_spatial),
            lr=self.learning_rate_spatial,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay
        )

        # Spatial Scheduler
        lr_scheduler_spatial = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer_spatial,
            num_warmup_steps=0,
            num_training_steps=max_training_steps,
        )

        unet_negation_all = unet_negation_spatial + unet_negation_temporal
        num_update_steps_per_epoch = len(train_dataloader)
        num_train_epochs = math.ceil(max_training_steps / num_update_steps_per_epoch)

        # Prepare callback
        step_callback = get_step_callback(
            overall_steps=max_training_steps,
            task="training",
            progress_callback=progress_callback
        )

        # Support mixed-precision training
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision_training else None

        # Standardize validation
        if validation_prompt is None:
            validation_prompt = []
        if validation_negative_prompt is None:
            validation_negative_prompt = []

        validation_prompts = [validation_prompt] if not isinstance(validation_prompt, list) else validation_prompt
        validation_negative_prompts = [validation_negative_prompt] if not isinstance(validation_negative_prompt, list) else validation_negative_prompt

        if validation_prompts and validation_negative_prompts:
            num_prompts = len(validation_prompts)
            num_negative_prompts = len(validation_negative_prompts)
            validation_negative_prompts = [
                validation_negative_prompts[i]
                if i < num_negative_prompts else validation_negative_prompts[-1]
                for i in range(num_prompts)
            ]

        # Train!
        total_batch_size = self.batch_size
        global_step = 0
        first_epoch = 0

        samples: List[List[Tuple[int, str, Image.Image]]] = []

        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                spatial_scheduler_lr = 0.0
                temporal_scheduler_lr = 0.0

                # Handle Lora Optimizers & Conditions
                optimizer_spatial.zero_grad(set_to_none=True)
                optimizer_temporal.zero_grad(set_to_none=True)

                mask_temporal_lora = not train_temporal
                mask_spatial_lora = not train_spatial

                mask_spatial_lora = True

                if self.random_null_text_ratio:
                    batch["text_prompt"] = [name if random.random() > self.random_null_text_ratio else "" for name in batch["text_prompt"]]

                # Data batch sanity check
                if epoch == first_epoch and step == 0:
                    # SANITY CHECK
                    pass

                # Convert videos to latent space            
                pixel_values = batch["pixel_values"].to(device)
                video_length = pixel_values.shape[2]
                bsz = pixel_values.shape[0]       

                # Sample a random timestep for each video
                timesteps = torch.randint(0, temporal_scheduler.config.num_train_timesteps, (bsz,), device=pixel_values.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                latents = tensor_to_vae_latent(pixel_values, vae)
                noise = sample_noise(latents, 0, use_offset_noise=self.use_offset_noise)
                target = noise         

                # Get the text embedding for conditioning
                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch["text_prompt"], 
                        max_length=tokenizer.model_max_length, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt"
                    ).input_ids.to(pixel_values.device)
                    encoder_hidden_states = text_encoder(prompt_ids)[0]

                with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                    if mask_spatial_lora:
                        loras = extract_lora_child_module(unet, target_replace_module=self.target_spatial_modules)
                        scale_loras(loras, 0.)
                        loss_spatial = None
                    else:
                        loras = extract_lora_child_module(unet, target_replace_module=self.target_spatial_modules)
                        scale_loras(loras, 1.0)
                        loras = extract_lora_child_module(unet, target_replace_module=self.target_temporal_modules)
                        if len(loras) > 0:
                            scale_loras(loras, 0.)

                        # Spatial LoRA Prediction
                        noisy_latents = temporal_scheduler.add_noise(latents, noise, timesteps)
                        noisy_latents_input, target_spatial, use_hflip = get_spatial_latents(
                            batch, 
                            self.random_hflip_img, 
                            False,
                            noisy_latents,
                            target,
                            timesteps,
                            spatial_scheduler
                        )

                        if use_hflip:
                            model_pred_spatial = unet(
                                noisy_latents_input,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states
                            ).sample
                            loss_spatial = F.mse_loss(
                                model_pred_spatial[:, :, 0, :, :].float(),
                                target_spatial[:, :, 0, :, :].float(),
                                reduction="mean"
                            )
                        else:
                            model_pred_spatial = unet(
                                noisy_latents_input.unsqueeze(2),
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states
                            ).sample
                            loss_spatial = F.mse_loss(
                                model_pred_spatial[:, :, 0, :, :].float(),
                                target_spatial.float(),
                                reduction="mean"
                            )

                    if mask_temporal_lora:
                        loras = extract_lora_child_module(unet, target_replace_module=self.target_temporal_modules)
                        scale_loras(loras, 0.)
                        loss_temporal = None
                    else:
                        loras = extract_lora_child_module(unet, target_replace_module=self.target_temporal_modules)
                        scale_loras(loras, 1.0)

                        ### Temporal LoRA Prediction ###
                        noisy_latents = spatial_scheduler.add_noise(latents, noise, timesteps)
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                        loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        loss_temporal = create_ad_temporal_loss(model_pred, loss_temporal, target)

                    # Backpropagate
                    if not mask_spatial_lora:
                        scaler.scale(loss_spatial).backward(retain_graph=True)
                        scaler.step(optimizer_spatial)

                    if not mask_temporal_lora and train_temporal:
                        scaler.scale(loss_temporal).backward()
                        scaler.step(optimizer_temporal)

                    lr_scheduler_spatial.step()
                    spatial_scheduler_lr = lr_scheduler_spatial.get_lr()[0]
                    if lr_scheduler_temporal is not None:
                        lr_scheduler_temporal.step()
                        temporal_scheduler_lr = lr_scheduler_temporal.get_lr()[0]

                scaler.update()
                global_step += 1
                step_callback(True)

                # Save checkpoint
                if global_step % self.checkpointing_steps == 0:
                    logger.info(f"Saving checkpoint at global step {global_step}")
                    # We do this to prevent VRAM spiking / increase from the new copy
                    pipeline.to("cpu")

                    if train_spatial and lora_manager_spatial is not None:
                        lora_manager_spatial.save_lora_weights(
                            model=deepcopy(pipeline),
                            save_path=os.path.join(lora_dir, "spatial"),
                            step=global_step,
                            use_safetensors=True,
                            lora_rank=self.lora_rank,
                            lora_name=f"{name}_spatial"
                        )

                    if train_temporal and lora_manager_temporal is not None:
                        lora_manager_temporal.save_lora_weights(
                            model=deepcopy(pipeline),
                            save_path=os.path.join(lora_dir, "temporal"),
                            step=global_step,
                            use_safetensors=True,
                            lora_rank=self.lora_rank,
                            lora_name=f"{name}_temporal",
                            use_motion_lora_format=self.use_motion_lora_format
                        )

                    pipeline.to(device)

                # Periodic validation
                if (global_step % self.validation_steps == 0 or global_step == 1):
                    logger.info(f"Generating validation samples at global step {global_step}")
                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(self.seed)

                    with torch.cuda.amp.autocast(enabled=True):
                        unet.disable_gradient_checkpointing()
                        loras = extract_lora_child_module(
                            unet, 
                            target_replace_module=self.target_spatial_modules
                        )
                        scale_loras(loras, self.validation_spatial_scale)
                        with torch.no_grad():
                            unet.eval()
                            for idx, (prompt, negative_prompt) in enumerate(zip(validation_prompts, validation_negative_prompts)):
                                sample = pipeline(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    device=device,
                                    generator=generator,
                                    animation_frames=self.sample_frames,
                                    height=self.sample_height,
                                    width=self.sample_width,
                                    num_inference_steps=self.sample_steps,
                                    guidance_scale=self.sample_guidance_scale,
                                ).images
                                Video(sample).save(
                                    os.path.join(samples_dir, f"sample-{global_step}-{idx}.gif"),
                                    overwrite=True,
                                    rate=8.0
                                )
                                samples.append((global_step, prompt, sample))
                            unet.train()
                if global_step >= max_training_steps:
                    break
                unet.enable_gradient_checkpointing()
        return samples
