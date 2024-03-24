import torch

from typing import Optional, Union

from enfugue.util import logger
from enfugue.diffusion.util import (
    inject_state_dict,
    empty_cache,
    TorchDataTypeConverter
)
from pibble.util.numeric import human_size
from enfugue.diffusion.support.upscale.supir.sgm.util import instantiate_from_config # type: ignore

def get_sampler_classname(use_tiling: bool, use_dpmpp: bool) -> str:
    if use_tiling:
        sampler_class = "TiledRestoreDPMPP2MSampler" if use_dpmpp else "TiledRestoreEDMSampler"
    else:
        sampler_class = "RestoreDPMPP2MSampler" if use_dpmpp else "RestoreEDMSampler"
    return f"enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.sampling.{sampler_class}"

def get_supir(
    sdxl_ckpt: str,
    supir_ckpt: str,
    clip_path: str,
    use_tiling: bool=True,
    use_dpmpp: bool=False,
    cache_dir: Optional[str]=None,
    device: Optional[Union[str, torch.device]]="cpu",
    ae_dtype: Optional[Union[str, torch.dtype]]=torch.bfloat16,
    diffusion_dtype: Optional[Union[str, torch.dtype]]=torch.float16,
) -> torch.nn.Module:
    """
    Gets the SUPIR model.
    Benchmarks CPU/GPU RAM usage.
    """
    from omegaconf import OmegaConf
    if use_tiling:
        sampler_params = {"tile_size": 128, "tile_stride": 64}
    else:
        sampler_params = {}

    sampler_config = {
        "target": get_sampler_classname(use_tiling, use_dpmpp),
        "params": {
            **{
                "num_steps": 100,
                "restore_cfg": 4,
                "s_churn": 0,
                "s_noise": 1.003,
                "discretization_config": {
                    "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
                },
                "guider_config": {
                    "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.guiders.LinearCFG",
                    "params": {
                        "scale": 7.5,
                        "scale_min": 4
                    }
                }
            },
            **sampler_params
        }
    }

    config = {
        "target": "enfugue.diffusion.support.upscale.supir.models.SUPIR_model.SUPIRModel",
        "params": {
            "ae_dtype": None if ae_dtype is None else str(ae_dtype),
            "diffusion_dtype": None if diffusion_dtype is None else str(diffusion_dtype),
            "scale_factor": 0.13025,
            "disable_first_stage_autocast": True,
            "network_wrapper": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.wrappers.ControlWrapper",
            "denoiser_config": {
                "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.denoiser.DiscreteDenoiserWithControl",
                "params": {
                    "num_idx": 1000,
                    "weighting_config": {
                        "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting"
                    },
                    "scaling_config": {
                        "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.denoiser_scaling.EpsScaling"
                    },
                    "discretization_config": {
                        "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
                    }
                }
            },
            "control_stage_config": {
                "target": "enfugue.diffusion.support.upscale.supir.modules.SUPIR_v0.GLVControl",
                "params": {
                    "adm_in_channels": 2816,
                    "num_classes": "sequential",
                    "use_checkpoint": True,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [
                        4,
                        2
                    ],
                    "num_res_blocks": 2,
                    "channel_mult": [
                        1,
                        2,
                        4
                    ],
                    "num_head_channels": 64,
                    "use_spatial_transformer": True,
                    "use_linear_in_transformer": True,
                    "transformer_depth": [
                        1,
                        2,
                        10
                    ],
                    "context_dim": 2048,
                    "spatial_transformer_attn_type": "softmax-xformers",
                    "legacy": False,
                    "input_upscale": 1
                }
            },
            "network_config": {
                "target": "enfugue.diffusion.support.upscale.supir.modules.SUPIR_v0.LightGLVUNet",
                "params": {
                    "mode": "XL-base",
                    "project_type": "ZeroSFT",
                    "project_channel_scale": 2,
                    "adm_in_channels": 2816,
                    "num_classes": "sequential",
                    "use_checkpoint": True,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [
                        4,
                        2
                    ],
                    "num_res_blocks": 2,
                    "channel_mult": [
                        1,
                        2,
                        4
                    ],
                    "num_head_channels": 64,
                    "use_spatial_transformer": True,
                    "use_linear_in_transformer": True,
                    "transformer_depth": [
                        1,
                        2,
                        10
                    ],
                    "context_dim": 2048,
                    "spatial_transformer_attn_type": "softmax-xformers",
                    "legacy": False
                }
            },
            "conditioner_config": {
                "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.GeneralConditionerWithControl",
                "params": {
                    "emb_models": [
                        {
                            "is_trainable": False,
                            "input_key": "txt",
                            "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules.FrozenCLIPEmbedder",
                            "params": {
                                "layer": "hidden",
                                "layer_idx": 11,
                                "cache_dir": cache_dir,
                                "device": str(device)
                            }
                        },
                        {
                            "is_trainable": False,
                            "input_key": "txt",
                            "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder2",
                            "params": {
                                "arch": "ViT-bigG-14",
                                "version": clip_path,
                                "freeze": True,
                                "layer": "penultimate",
                                "always_return_pooled": True,
                                "legacy": False,
                            }
                        },
                        {
                            "is_trainable": False,
                            "input_key": "original_size_as_tuple",
                            "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules.ConcatTimestepEmbedderND",
                            "params": {
                                "outdim": 256
                            }
                        },
                        {
                            "is_trainable": False,
                            "input_key": "crop_coords_top_left",
                            "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules.ConcatTimestepEmbedderND",
                            "params": {
                                "outdim": 256
                            }
                        },
                        {
                            "is_trainable": False,
                            "input_key": "target_size_as_tuple",
                            "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules.ConcatTimestepEmbedderND",
                            "params": {
                                "outdim": 256
                            }
                        }
                    ]
                }
            },
            "first_stage_config": {
                "target": "enfugue.diffusion.support.upscale.supir.sgm.models.autoencoder.AutoencoderKLInferenceWrapper",
                "params": {
                    "ckpt_path": None,
                    "embed_dim": 4,
                    "monitor": "val/rec_loss",
                    "ddconfig": {
                        "attn_type": "vanilla-xformers",
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [
                            1,
                            2,
                            4,
                            4
                        ],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    }
                }
            },
            "sampler_config": sampler_config,
        }
    }
    logger.debug("Instantiating SUPIR model.")
    model = instantiate_from_config(OmegaConf.create(config))
    logger.debug(f"Injecting SDXL checkpoint state dictionary from {sdxl_ckpt}")
    unexpected = inject_state_dict(sdxl_ckpt, model, strict=False)
    if unexpected:
        logger.info(f"Discarded SDXL checkpoint weights {unexpected}")
    logger.debug(f"Injecting SUPIR checkpoint state dictionary from {supir_ckpt}")
    unexpected = inject_state_dict(supir_ckpt, model, strict=False)
    if unexpected:
        logger.info(f"Discarded SUPIR checkpoint weights {unexpected}")
    if ae_dtype is not None:
        logger.debug(f"Casting first stage model to {ae_dtype}")
        model.first_stage_model.to(dtype=TorchDataTypeConverter.convert(ae_dtype))
    if diffusion_dtype is not None:
        logger.debug(f"Casting diffusion model to {diffusion_dtype}")
        model.model.to(dtype=TorchDataTypeConverter.convert(diffusion_dtype))

    model = model.to(device=device)
    empty_cache()
    return model
