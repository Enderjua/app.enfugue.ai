from __future__ import annotations

from contextlib import contextmanager

from typing import Iterator, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image

from enfugue.util.log import logger
from enfugue.diffusion.support.model import (
    SupportModel,
    SupportModelPipeline,
    SupportModelProcessor
)

__all__ = [
    "ImageAnimatorProcessor",
    "ImageAnimator"
]

class ImageAnimatorProcessor(SupportModelProcessor):
    """
    Uses an image-to-video LVDM pipeline.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        use_interp: bool=False
    ) -> None:
        """
        Holds the NN module and provides a callable
        """
        self.model = model
        self.use_interp = use_interp

    def __call__(
        self,
        images: Union[Image, Tuple[Image, ...], List[Image]],
        prompt: str,
        steps: int=50,
        cfg_scale: float=7.5,
        eta: float=1.0,
        fs: int=5,
        num_frames: int=16,
        temperature: float=1.0,
        seed: Optional[int]=None,
        frame_window_size: Optional[int]=None,
        frame_window_stride: Optional[int]=None,
        mask: Optional[Union[Image, Tuple[Image, ...], List[Image]]]=None,
    ) -> List[Image]:
        """
        Executes the pipeline
        """
        import torch
        from enfugue.util import fit_image, scale_image
        from enfugue.diffusion.util import image_to_tensor, tensor_to_image, seed_all
        from enfugue.diffusion.animate.dynamicrafter.scripts.evaluation.funcs import batch_ddim_sampling

        # Seed everything
        if seed is None:
            from random import randint
            seed = randint(0x1000000, 0xFFFFFFF)
        seed_all(seed)

        # Get model channels
        c = self.model.model.diffusion_model.out_channels

        # Get device/dtypes
        device = self.model.device
        first_stage_dtype = self.model.first_stage_model.dtype
        model_dtype = self.model.model.dtype

        # Standardize images
        if not isinstance(images, tuple) and not isinstance(images, list):
            images = [images]

        # Standardize mask(s)
        if mask is not None and not isinstance(mask, tuple) and not isinstance(mask, list):
            mask = [mask]

        # Memoize real width/height for later
        width, height = images[0].size

        # Scale image to nearest
        images = [
            scale_image(image, nearest=64).convert("RGB")
            for image in images
        ]

        w, h = images[0].size
        w = w // 8
        h = h // 8

        if mask is not None:
            mask = [
                m.resize((w, h))
                for m in mask
            ]

        # Make sure frame sizes are correct
        if frame_window_size and frame_window_stride:
            frame_offset = ((num_frames - frame_window_size) % frame_window_stride)
            if frame_offset:
                logger.warning(f"Number of frames ({num_frames}) minus frame window size ({frame_window_size}) must be divisible be frame window stride ({frame_window_stride}), adding {frame_offset} frame(s).")
                num_frames += frame_offset

        # Assign to model
        self.model.image_size = [h, w]

        # Disable grad
        with torch.no_grad(), torch.autocast(device.type, dtype=model_dtype):
            # Get total condition
            condition = torch.zeros((1, c, num_frames, h, w), device=device, dtype=model_dtype)
            images = images[:num_frames]
            num_images = len(images)

            # Convert images to tensor in range (-1, 1)
            images = [
                image_to_tensor(image, lower=-1.0)
                for image in images
            ]

            # Convert mask(s)
            if mask is not None:
                num_masks = len(mask)
                mask = torch.cat([
                    torch.mean(image_to_tensor(m), dim=1).unsqueeze(1).unsqueeze(2)
                    for m in mask
                ], dim=2)
                if num_masks < num_frames:
                    mask = torch.cat([mask, mask[:, :, -1:, :, :].repeat(1, 1, num_frames-num_masks, 1, 1)], dim=2)
                mask = torch.where(mask < 1.0, 0.0, 1.0).to(device=device, dtype=model_dtype)

            # Encode in VAE
            encoded_images = [
                self.model.encode_first_stage(
                    image.to(device=device, dtype=first_stage_dtype)
                ).to(dtype=model_dtype).unsqueeze(2)
                for image in images
            ]

            # Assign to condition
            if self.use_interp:
                condition[:, :, :1, :, :] = encoded_images[0]
                condition[:, :, -1:, :, :] = encoded_images[-1]
            else:
                condition[:, :, :num_images, :, :] = torch.cat(encoded_images, dim=2)
                if num_images < num_frames:
                    condition[:, :, num_images:, :, :] = encoded_images[-1].repeat(1, 1, num_frames-num_images, 1, 1)

            # Get embeddings
            text_emb = self.model.get_learned_conditioning([prompt])
            cond_images = self.model.embedder(
                images[0].to(device=device, dtype=model_dtype)
            )
            image_emb = self.model.image_proj_model(cond_images)
            image_text_cond = torch.cat([text_emb, image_emb], dim=1)

            # Get FS tensor
            fs = torch.tensor([fs], dtype=torch.long, device=device)

            # Combine to inputs
            cond = {
                "c_crossattn": [image_text_cond],
                "fs": fs,
                "c_concat": [condition]
            }

            # Run inference
            batch_samples = batch_ddim_sampling(
                self.model,
                cond,
                (1, c, num_frames, h, w),
                mask=mask,
                x0=condition.clone(),
                n_samples=1,
                ddim_steps=steps,
                ddim_eta=eta,
                cfg_scale=cfg_scale,
                precision=16 if model_dtype is torch.float16 else None,
                frame_window_size=frame_window_size,
                frame_window_stride=frame_window_stride,
                temperature=temperature,
            )[0] # Now B C F H W

            # Return to images
            return [
                tensor_to_image(
                    batch_samples[:, :, i, :, :],
                    lower=-1.0
                ).resize((width, height))
                for i in range(num_frames)
            ]

class ImageAnimator(SupportModel):
    """
    Maps image-to-video models to callables
    """
    DYNAMICRAFTER_1024_PATH = "https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt"
    DYNAMICRAFTER_512_PATH = "https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt"
    DYNAMICRAFTER_512_INTERP_PATH = "https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/resolve/main/model.ckpt"

    @property
    def dynamicrafter_512_path(self) -> str:
        """
        Downsloads the dynamicrafter 512 model
        """
        return self.get_model_file(
            self.DYNAMICRAFTER_512_PATH,
            filename="dynamicrafter_512.ckpt",
            extensions=[".bin", ".pt", ".pth", ".ckpt", ".safetensors"]
        )

    @property
    def dynamicrafter_512_interp_path(self) -> str:
        """
        Downsloads the dynamicrafter 512 interp model
        """
        return self.get_model_file(
            self.DYNAMICRAFTER_512_INTERP_PATH,
            filename="dynamicrafter_512_interp.ckpt",
            extensions=[".bin", ".pt", ".pth", ".ckpt", ".safetensors"]
        )

    @property
    def dynamicrafter_1024_path(self) -> str:
        """
        Downsloads the dynamicrafter 1024 model
        """
        return self.get_model_file(
            self.DYNAMICRAFTER_1024_PATH,
            filename="dynamicrafter_1024.ckpt",
            extensions=[".bin", ".pt", ".pth", ".ckpt", ".safetensors"]
        )

    def dynamicrafter(self, *args: Any, **kwargs: Any) -> SupportModelPipeline:
        """
        Gets a re-useable model
        """
        return self.get_pipeline("dynamicrafter_processor", *args, **kwargs)

    @contextmanager
    def dynamicrafter_processor(
        self,
        model: Literal["512","1024","512-interp"]="1024",
    ) -> Iterator[ImageAnimatorProcessor]:
        """
        Gets the specified dynamicrafter processor
        """
        import torch
        from enfugue.diffusion.animate.dynamicrafter.helper import get_model
        use_interp = False
        if model == "512":
            model_file = self.dynamicrafter_512_path
        elif model == "512-interp":
            use_interp = True
            model_file = self.dynamicrafter_512_interp_path
        elif model == "1024":
            model_file = self.dynamicrafter_1024_path
        else:
            raise ValueError(f"Unknown model '{model}'") # type: ignore

        with self.context():
            model = get_model(
                model_path=model_file,
                model_size="1024" if model == "1024" else "512",
                model_dtype=self.dtype
            )
            model = model.to(device=self.device)
            processor = ImageAnimatorProcessor(model, use_interp)
            yield processor
            del processor
            del model
