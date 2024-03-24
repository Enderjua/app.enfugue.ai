from __future__ import annotations
import os

from typing import Any, Iterator, Literal, Optional, Callable, TYPE_CHECKING

from contextlib import contextmanager

from PIL import Image

from enfugue.util import timed
from enfugue.diffusion.constants import *
from enfugue.diffusion.util import ComputerVision
from enfugue.diffusion.support.model import SupportModel, SupportModelProcessor

if TYPE_CHECKING:
    from torch import device as Device, dtype as DType, Tensor
    from realesrgan import RealESRGANer
    from enfugue.diffusion.support.upscale.gfpganer import GFPGANer  # type: ignore[attr-defined]
    from enfugue.diffusion.support.upscale.ccsr.model.ccsr_stage2 import ControlLDM  # type: ignore[attr-defined]

__all__ = ["Upscaler"]

class ESRGANProcessor(SupportModelProcessor):
    """
    Holds a reference to the esrganer and provides a callable
    """
    def __init__(self, esrganer: RealESRGANer, **kwargs: Any) -> None:
        super(ESRGANProcessor, self).__init__(**kwargs)
        self.esrganer = esrganer

    def __call__(self, image: Image.Image, outscale: int = 2) -> Image.Image:
        """
        Upscales an image
        """
        return ComputerVision.revert_image(
            self.esrganer.enhance(
                ComputerVision.convert_image(image),
                outscale=outscale
            )[0]
        )

class GFPGANProcessor(SupportModelProcessor):
    """
    Holds a reference to the gfpganer and provides a callable
    """
    def __init__(self, gfpganer: GFPGANer, **kwargs: Any) -> None:
        super(GFPGANProcessor, self).__init__(**kwargs)
        self.gfpganer = gfpganer

    def __call__(self, image: Image.Image, outscale: int = 2) -> Image.Image:
        """
        Upscales an image
        GFPGan is fixed at x4 so this fixes the scale here
        """
        result = ComputerVision.revert_image(
            self.gfpganer.enhance(
                ComputerVision.convert_image(image),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )[2]
        )
        width, height = result.size
        multiplier = outscale / 4
        return result.resize((int(width * multiplier), int(height * multiplier)))

class CCSRProcessor(SupportModelProcessor):
    """
    Holds a reference to the CCSR LDM and provides a callable
    """
    def __init__(self, ccsr: ControlLDM, **kwargs: Any) -> None:
        super(CCSRProcessor, self).__init__(**kwargs)
        self.ccsr = ccsr

    def __call__(
        self,
        image: Image.Image,
        num_steps: int=45,
        strength: float=1.0,
        tile_diffusion_size: Optional[int]=512,
        tile_diffusion_stride: Optional[int]=256,
        tile_vae_decode_size: Optional[int]=128,
        tile_vae_encode_size: Optional[int]=512,
        color_fix_type: Optional[Literal["wavelet", "adain"]]="adain",
        t_min: float=0.3333,
        t_max: float=0.6667,
        seed: Optional[int]=None,
        positive_prompt: str="",
        negative_prompt: str="",
        cfg_scale: float=1.0,
        outscale: int=2,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
    ) -> Image.Image:
        """
        Upscales an image using CCSR
        """
        import torch
        import numpy as np
        from einops import rearrange
        from math import ceil, floor
        from enfugue.util import get_step_callback, logger
        from enfugue.diffusion.support.upscale.ccsr.utils.image import auto_resize # type: ignore
        from enfugue.diffusion.support.upscale.ccsr.model.q_sampler import SpacedSampler # type: ignore

        # seed
        if seed is not None:
            import pytorch_lightning as pl
            pl.seed_everything(seed)

        # Resize bicubic first
        image = image.convert("RGB").resize(
            tuple(ceil(x*outscale) for x in image.size),
            Image.BICUBIC
        )
        # Get the condition
        condition = auto_resize(image, 512 if not tile_diffusion_size else tile_diffusion_size)
        condition = condition.resize(
            tuple((x//64+1)*64 for x in condition.size),
            Image.LANCZOS
        )
        condition = torch.tensor(
            np.array(condition) / 255.0,
            dtype=torch.float32,
            device=self.ccsr.device
        ).unsqueeze(0)
        condition = rearrange(condition.clamp_(0, 1), "n h w c -> n c h w").contiguous()
        # Create noise
        height, width = condition.shape[-2:]
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.ccsr.device, dtype=torch.float32)
        # Adjust control scale in model
        self.ccsr.control_scales = [strength] * 13
        # Instantiate sampler
        sampler = SpacedSampler(self.ccsr, var_type="fixed_small")
        num_decode_steps = 1
        # Set tiling
        if tile_vae_decode_size and tile_vae_encode_size and max(height // 8, width // 8) > 22 + tile_vae_decode_size:
            num_decode_height_tiles = ceil((height // 8 - 22) / tile_vae_decode_size)
            num_decode_width_tiles = ceil((width // 8 - 22) / tile_vae_decode_size)
            num_decode_steps = max(1, num_decode_height_tiles) * max(1, num_decode_width_tiles)
            self.ccsr._init_tiled_vae(
                encoder_tile_size=tile_vae_encode_size,
                decoder_tile_size=tile_vae_decode_size
            )
        # Gather sampler arguments
        sampler_kwargs = {
            "steps": num_steps,
            "t_max": t_max,
            "t_min": t_min,
            "shape": shape,
            "cond_img": condition,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "x_T": x_T,
            "cfg_scale": cfg_scale,
            "color_fix_type": "none" if not color_fix_type else color_fix_type
        }
        if tile_diffusion_size and tile_diffusion_stride and max(height, width) > tile_diffusion_size:
            sample_windows = sampler.get_sliding_windows(
                height // 8,
                width // 8,
                tile_diffusion_size // 8,
                tile_diffusion_stride // 8
            )
            num_sample_steps = len(sample_windows) * (1 + num_steps)
            sampler_kwargs["tile_size"] = tile_diffusion_size
            sampler_kwargs["tile_stride"] = tile_diffusion_stride
            execute_sampler = sampler.sample_with_tile_ccsr
        else:
            num_sample_steps = num_steps
            execute_sampler = sampler.sample_ccsr

        # Get step callback
        num_sample_steps = floor(num_sample_steps * (t_max - t_min))
        total_steps = num_decode_steps + num_sample_steps
        logger.debug(f"Calculated total steps to be {total_steps}: {num_sample_steps} sampling step(s) + {num_decode_steps} decoding step(s)")
        step_complete = get_step_callback(total_steps, progress_callback=progress_callback)

        # Execute sampler
        samples = execute_sampler(step_complete=step_complete, **sampler_kwargs)

        # Return to image
        samples = samples.clamp(0, 1)
        samples = (rearrange(samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        return Image.fromarray(samples[0])

class FaceRestoreProcessor(SupportModelProcessor):
    """
    Holds a reference to the gfpganer and provides a callable
    """
    def __init__(self, gfpganer: GFPGANer, **kwargs: Any) -> None:
        super(FaceRestoreProcessor, self).__init__(**kwargs)
        self.gfpganer = gfpganer

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Upscales an image
        """
        return ComputerVision.revert_image(
            self.gfpganer.enhance(
                ComputerVision.convert_image(image),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )[2]
        )

class SUPIRProcessor(SupportModelProcessor):
    """
    Processes an image using SUPIR.
    Also provides helper methods for using an LLM get get tiled upscale prompts.
    """
    DEFAULT_POSITIVE_PROMPT = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo, realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    DEFAULT_NEGATIVE_PROMPT = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"

    def __init__(self, model: SUPIRModel, device: Device, dtype: DType, **kwargs: Any) -> None:
        super(SUPIRProcessor, self).__init__(**kwargs)
        self.model = model
        self.device = device
        self.dtype = dtype

    def get_upscaled_tensor(
        self,
        image: Image.Image,
        outscale: int=2,
        gamma_correction: float=1.0,
    ) -> Tensor:
        """
        Converts an image to a tensor with gamma correction and upscaling.
        """
        import torch
        import numpy as np
        from enfugue.diffusion.support.upscale.supir.util import upscale_image, HWC3

        input_image = ComputerVision.convert_image(image)
        input_image = HWC3(input_image)
        input_image = upscale_image(input_image, outscale, unit_resolution=32, min_size=1024)

        lq = np.array(input_image) / 255.0
        lq = np.power(lq, gamma_correction)
        lq *= 255.0
        lq = lq.round().clip(0, 255).astype(np.uint8)
        lq = lq / 255 * 2 - 1
        lq = torch.tensor(lq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)[:, :3, :, :]
        return lq

    def finalize(self, lq: Tensor, gamma_correction: float=1.0) -> Image.Image:
        """
        Finalizes a tensor back to an image, with gamma correction
        """
        import numpy as np
        lq = (lq[0].permute(1,2,0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
        lq = lq / 255.0
        lq = np.power(lq, gamma_correction)
        lq = lq * 255.0
        lq = lq.round().clip(0, 255).astype(np.uint8)
        return ComputerVision.revert_image(lq)

    def __call__(
        self,
        image: Image.Image,
        captions: List[str] = [],
        outscale: int=2,
        num_steps: int=100,
        restoration_scale: float=-1.0,
        s_churn: int=5,
        s_noise: float=1.003,
        prompt: Optional[str]=None,
        negative_prompt: Optional[str]=None,
        tile_diffusion_size: Optional[int]=1024,
        tile_diffusion_stride: Optional[int]=512,
        tile_vae_decode_size: Optional[int]=1024,
        tile_vae_encode_size: Optional[int]=1024,
        color_fix_type: Optional[Literal["wavelet", "adain"]]="wavelet",
        use_linear_cfg: bool=True,
        cfg_scale: float=7.5, # 4.0
        cfg_scale_start: float=4.0, # 1.0
        use_linear_control_scale: bool=False,
        control_scale: float=1.0,
        control_scale_start: float=0.0,
        gamma_correction: float=1.0,
        num_samples: int=1,
        use_dpmpp: bool=False,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
        seed: Optional[int]=None,
    ) -> Image.Image:
        """
        Upscales an image using SUPIR
        """
        import numpy as np
        from einops import rearrange
        from enfugue.diffusion.support.upscale.supir.helper import get_sampler_classname
        from enfugue.util import get_step_callback, logger, sliding_window_count
        if prompt is None:
            prompt = self.DEFAULT_POSITIVE_PROMPT
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT
        lq = self.get_upscaled_tensor(image, outscale=outscale, gamma_correction=gamma_correction)
        b, c, h, w = lq.shape
        if tile_vae_encode_size and tile_vae_decode_size:
            self.model.init_tile_vae(
                encoder_tile_size=tile_vae_encode_size,
                decoder_tile_size=tile_vae_decode_size // 8,
                device=self.device,
            )
        if not captions:
            captions = [""]
        else:
            captions = [captions]

        if tile_diffusion_size and tile_diffusion_stride:
            diffusion_steps = (num_steps - 1) * sliding_window_count(
                height=h,
                width=w,
                tile_size=tile_diffusion_size,
                tile_stride=tile_diffusion_stride,
            )
            self.model.sampler_config.target = get_sampler_classname(
                use_tiling=True,
                use_dpmpp=use_dpmpp
            )
            self.model.sampler_config.params.tile_size = tile_diffusion_size // 8
            self.model.sampler_config.params.tile_stride = tile_diffusion_stride // 8
        else:
            try:
                del self.model.sampler_config.params.tile_size
            except:
                pass
            try:
                del self.model.sampler_config.params.tile_stride
            except:
                pass
            diffusion_steps = num_steps - 1
            self.model.sampler_config.target = get_sampler_classname(
                use_tiling=False,
                use_dpmpp=use_dpmpp
            )

        step_complete = get_step_callback(diffusion_steps, progress_callback=progress_callback)
        samples = self.model.batchify_sample(
            lq.to(device=self.device, dtype=self.dtype),
            captions,
            num_steps=num_steps,
            restoration_scale=restoration_scale,
            s_churn=s_churn,
            s_noise=s_noise,
            cfg_scale=cfg_scale,
            control_scale=control_scale,
            num_samples=num_samples,
            p_p=prompt,
            n_p=negative_prompt,
            color_fix_type={"wavelet": "Wavelet", "adain": "AdaIn"}.get(color_fix_type, "None"),
            use_linear_CFG=use_linear_cfg,
            use_linear_control_scale=use_linear_control_scale,
            cfg_scale_start=cfg_scale_start,
            control_scale_start=control_scale_start,
            sampler_callback=lambda: step_complete(True),
            seed=-1 if seed is None else seed
        )
        x_samples = (rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
        return ComputerVision.revert_image(x_samples[0])

class APISRProcessor(SupportModelProcessor):
    """
    A small wrapper around the APISR RRDB GAN
    """
    def __init__(self, model: RRDBNet, device: Device, dtype: DType) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype

    def __call__(self, image: Image.Image, outscale: int = 2) -> Image.Image:
        """
        Upscales using repeated 2x scaling.
        """
        import torch
        import torch.nn.functional as F
        from enfugue.diffusion.util import image_to_tensor, tensor_to_image
        with torch.autocast(self.device.type, dtype=self.dtype):
            image = image_to_tensor(image.convert("RGB")).to(device=self.device, dtype=self.dtype)
            _, _, h, w = image.shape
            final_h = int(h * outscale)
            final_w = int(w * outscale)
            image = F.interpolate(image, size=((h//8)*8, (w//8)*8), mode="bicubic")
            while outscale > 1:
                image = self.model(image)
                outscale /= 2
            return tensor_to_image(
                F.interpolate(image, size=(final_h, final_w), mode="bicubic")
            )

class Upscaler(SupportModel):
    """
    The upscaler user ESRGAN or GFGPGAN for up to 4x upscale
    """
    APISR_RRDB_PATH = "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth"
    ESRGAN_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ESRGAN_ANIME_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    
    GFPGAN_PATH = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    GFPGAN_DETECTION_PATH = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    GFPGAN_PARSENET_PATH = "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

    CCSR_CKPT_PATH = "https://huggingface.co/benjamin-paine/ccsr/resolve/main/real-world_ccsr.ckpt"

    SUPIR_F_CKPT_PATH = "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0F.ckpt"
    SUPIR_Q_CKPT_PATH = "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.fp16.safetensors"
    SUPIR_Q_MERGED_CKPT_PATH = "https://huggingface.co/benjaminpaine/SUPIR/resolve/main/supir-v0q-merged.fp16.safetensors"
    OPEN_CLIP_PATH = "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin"

    def get_upsampler(
        self,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        anime: bool = False
    ) -> RealESRGANer:
        """
        Gets the appropriate upsampler
        """
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if anime:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = self.get_model_file(self.ESRGAN_ANIME_PATH)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = self.get_model_file(self.ESRGAN_PATH)

        return RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            device=self.device,
            half=self.dtype is torch.float16,
        )

    @contextmanager
    def esrgan(
        self,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        anime: bool = False,
    ) -> Iterator[SupportModelProcessor]:
        """
        Does a simple upscale
        """
        with self.context():
            esrganer = self.get_upsampler(
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                anime=anime
            )
            processor = ESRGANProcessor(esrganer)
            yield processor
            del processor
            del esrganer

    @contextmanager
    def face_restore(self) -> Iterator[SupportModelProcessor]:
        """
        Only does face enhancement
        """
        with self.context():
            from enfugue.diffusion.support.upscale.gfpgan import GFPGANer  # type: ignore[attr-defined]
            model_path = self.get_model_file(self.GFPGAN_PATH)
            detection_model_path = self.get_model_file(self.GFPGAN_DETECTION_PATH)
            parse_model_path = self.get_model_file(self.GFPGAN_PARSENET_PATH)

            gfpganer = GFPGANer(
                model_path=model_path,
                detection_model_path=detection_model_path,
                parse_model_path=parse_model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                device=self.device,
                bg_upsampler=None
            )

            processor = FaceRestoreProcessor(gfpganer)
            yield processor
            del processor
            del gfpganer

    @contextmanager
    def gfpgan(
        self,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
    ) -> Iterator[SupportModelProcessor]:
        """
        Does an upscale with face enhancement
        """
        with self.context():
            from enfugue.diffusion.support.upscale.gfpgan import GFPGANer  # type: ignore[attr-defined]
            model_path = self.get_model_file(self.GFPGAN_PATH)
            detection_model_path = self.get_model_file(self.GFPGAN_DETECTION_PATH)
            parse_model_path = self.get_model_file(self.GFPGAN_PARSENET_PATH)

            gfpganer = GFPGANer(
                model_path=model_path,
                detection_model_path=detection_model_path,
                parse_model_path=parse_model_path,
                upscale=4,
                arch="clean",
                channel_multiplier=2,
                device=self.device,
                bg_upsampler=self.get_upsampler(
                    tile=tile,
                    tile_pad=tile_pad,
                    pre_pad=pre_pad
                )
            )

            processor = GFPGANProcessor(gfpganer)
            yield processor
            del processor
            del gfpganer

    @contextmanager
    def ccsr(self) -> Iterator[SupportModelProcessor]:
        """
        Does an upscale using CCSR (content consistent super-resolution)
        """
        with self.context():
            from enfugue.diffusion.support.upscale.ccsr.stage_2 import get_model # type: ignore
            from enfugue.diffusion.util.torch_util import load_state_dict
            ccsr_ckpt_path = self.get_model_file(self.CCSR_CKPT_PATH)
            state_dict = load_state_dict(ccsr_ckpt_path)
            ccsr_model = get_model(state_dict)
            del state_dict
            ccsr_model.freeze()
            ccsr_model.to(self.device)
            processor = CCSRProcessor(ccsr_model)
            yield processor
            del processor
            del ccsr_model

    @contextmanager
    def apisr(self) -> Iterator[SupportModelProcessor]:
        """
        Does an upscale using APISR (Anime Production Inspired Super-Resolution)
        """
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from enfugue.diffusion.util import load_state_dict

        model_path = self.get_model_file(self.APISR_RRDB_PATH)

        with self.context():
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=2
            )
            state_dict = load_state_dict(model_path)["model_state_dict"]
            model.load_state_dict(state_dict)
            del state_dict
            model = model.eval()
            model = model.to(dtype=self.dtype)
            model = model.to(device=self.device)
            processor = APISRProcessor(
                model=model,
                device=self.device,
                dtype=self.dtype
            )
            yield processor
            del processor
            del model

    @contextmanager
    def supir(
        self,
        sdxl_path: str=DEFAULT_SDXL_MODEL
    ) -> Iterator[SupportModelProcessor]:
        """
        Does an upscale with SUPIR (Scaling UP Image Restoration)
        """
        with self.context():
            import torch
            from enfugue.diffusion.support.upscale.supir.helper import get_supir
            supir_ckpt = self.get_model_file(
                self.SUPIR_Q_CKPT_PATH,
                check_size=False
            )
            sdxl_ckpt = self.get_model_file(
                sdxl_path,
                directory=self.kwargs.get("checkpoints_dir", os.path.join(self.root_dir, "checkpoint")),
            )
            clip_ckpt = self.get_model_file(
                self.OPEN_CLIP_PATH,
                directory=self.kwargs.get("clip_dir", os.path.join(self.root_dir, "clip"))
            )
            with timed(task="loading model"):
                model = get_supir(
                    sdxl_ckpt,
                    supir_ckpt,
                    clip_ckpt,
                    cache_dir=self.kwargs.get("cache_dir", os.path.join(self.root_dir, "cache")),
                    device=self.device,
                )
            processor = SUPIRProcessor(model=model, device=self.device, dtype=torch.bfloat16)
            yield processor
            del processor
            del model

    def __call__(
        self,
        method: Literal["esrgan", "esrganime", "gfpgan", "ccsr"],
        image: Image.Image,
        outscale: int = 2,
        **kwargs: Any
    ) -> Image:
        """
        Performs one quick upscale
        """
        if method == "esrgan":
            context = self.esrgan
        elif method == "esrganime":
            context = self.esrgan # type: ignore
            kwargs["anime"] = True
        elif method == "gfpgan":
            context = self.gfpgan # type: ignore
        elif method == "ccsr":
            context = self.ccsr # type: ignore
        else:
            raise ValueError(f"Unknown upscale method {method}") # type: ignore[unreachable]

        with context(**kwargs) as processor:
            return processor(image, outscale=outscale) # type: ignore
