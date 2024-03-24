from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from omegaconf import OmegaConf
from enfugue.diffusion.animate.dynamicrafter.utils import instantiate_from_config

if TYPE_CHECKING:
    import torch

CONFIG_BASE = """
target: enfugue.diffusion.animate.dynamicrafter.lvdm.models.ddpm3d.LatentVisualDiffusion
params:
    rescale_betas_zero_snr: true
    parameterization: "v"
    linear_start: 0.00085
    linear_end: 0.012
    num_timesteps_cond: 1
    timesteps: 1000
    first_stage_key: video
    cond_stage_key: caption
    cond_stage_trainable: false
    conditioning_key: hybrid
    channels: 4
    scale_by_std: false
    scale_factor: 0.18215
    use_ema: false
    uncond_type: 'empty_seq'
    use_dynamic_rescale: true
    fps_condition_type: 'fps'
    perframe_ae: true

    unet_config:
        target: enfugue.diffusion.animate.dynamicrafter.lvdm.modules.networks.openaimodel3d.UNetModel
        params:
            in_channels: 8
            out_channels: 4
            model_channels: 320
            attention_resolutions:
                - 4
                - 2
                - 1
            num_res_blocks: 2
            channel_mult:
                - 1
                - 2
                - 4
                - 4
            dropout: 0.1
            num_head_channels: 64
            transformer_depth: 1
            context_dim: 1024
            use_linear: true
            use_checkpoint: false
            temporal_conv: true
            temporal_attention: true
            temporal_selfatt_only: true
            use_relative_position: false
            use_causal_attention: false
            temporal_length: 16
            addition_attention: true
            image_cross_attention: true
            fs_condition: true

    first_stage_config:
        target: enfugue.diffusion.animate.dynamicrafter.lvdm.models.autoencoder.AutoencoderKL
        params:
            embed_dim: 4
            monitor: val/rec_loss
            ddconfig:
                double_z: true
                z_channels: 4
                resolution: 256
                in_channels: 3
                out_ch: 3
                ch: 128
                ch_mult:
                    - 1
                    - 2
                    - 4
                    - 4
                num_res_blocks: 2
                attn_resolutions: []
                dropout: 0.0
            lossconfig:
                target: torch.nn.Identity

    cond_stage_config:
        target: enfugue.diffusion.animate.dynamicrafter.lvdm.modules.encoders.condition.FrozenOpenCLIPEmbedder
        params:
            freeze: true
            layer: "penultimate"

    img_cond_stage_config:
        target: enfugue.diffusion.animate.dynamicrafter.lvdm.modules.encoders.condition.FrozenOpenCLIPImageEmbedderV2
        params:
            freeze: true

    image_proj_stage_config:
        target: enfugue.diffusion.animate.dynamicrafter.lvdm.modules.encoders.resampler.Resampler
        params:
            dim: 1024
            depth: 4
            dim_head: 64
            heads: 12
            num_queries: 16
            embedding_dim: 1280
            output_dim: 1024
            ff_mult: 4
            video_length: 16
"""

def get_config(
    model_size: Literal["1024", "512"],
    model_dtype: Optional[torch.dtype]=None,
) -> OmegaConf:
    """
    Gets a configuration and setsq it for a specific model.
    """
    import torch
    conf = OmegaConf.create(CONFIG_BASE)
    if model_size == "1024":
        conf.params.base_scale = 0.3
        conf.params.image_size = [72, 128]
        conf.params.unet_config.params.default_fs = 10
    elif model_size == "512":
        conf.params.base_scale = 0.7
        conf.params.image_size = [40, 64]
        conf.params.unet_config.params.default_fs = 24
    else:
        raise ValueError(f"Unsupported model size {model_size}") # type: ignore
    if model_dtype is torch.float16:
        conf.params.unet_config.params.use_fp16 = True
    return conf

def get_model(
    model_path: str,
    model_size: Literal["1024", "512"],
    model_dtype: Optional[torch.dtype]=None,
) -> nn.Module:
    """
    Instantiates from a module.
    """
    import torch
    from enfugue.diffusion.util import load_state_dict
    config = get_config(
        model_size=model_size,
        model_dtype=model_dtype
    )
    model = instantiate_from_config(config)
    state_dict = load_state_dict(model_path)
    model.load_state_dict(state_dict)
    if model_dtype is not None:
        model.to(dtype=model_dtype)
        if model_dtype is torch.float16:
            model.first_stage_model.to(torch.bfloat16)
    model.eval()
    return model
