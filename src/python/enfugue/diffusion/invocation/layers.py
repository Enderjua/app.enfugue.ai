from __future__ import annotations

import inspect

from contextlib import contextmanager, ExitStack
from datetime import datetime

from dataclasses import (
    dataclass,
    asdict,
    field,
)

from typing import (
    Optional,
    Dict,
    Any,
    Union,
    Tuple,
    List,
    Callable,
    Iterator,
    Callable,
    Optional,
    Literal,
    TYPE_CHECKING,
)
from math import floor
from random import randint

from enfugue.util import (
    logger,
    fit_image,
    get_frames_or_image,
    get_frames_or_image_from_file,
    dilate_erode,
    redact_images_from_metadata,
    redact_for_log,
)

from enfugue.diffusion.constants import *

from pibble.api.exceptions import BadRequestError

if TYPE_CHECKING:
    from PIL.Image import Image
    from enfugue.diffusion.manager import DiffusionPipelineManager
    from enfugue.util import IMAGE_FIT_LITERAL, IMAGE_ANCHOR_LITERAL

__all__ = ["LayeredInvocation"]

@dataclass
class LayeredInvocation:
    """
    A serializable class holding all vars for an invocation
    """
    # Dimensions, required
    width: int
    height: int
    # Model args
    model: Optional[str]=None
    refiner: Optional[str]=None
    inpainter: Optional[str]=None
    vae: Optional[str]=None
    refiner_vae: Optional[str]=None
    inpainter_vae: Optional[str]=None
    lora: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]]=None
    lycoris: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]]=None
    inversion: Optional[Union[str, List[str]]]=None
    ip_adapter_model: Optional[IP_ADAPTER_LITERAL]=None
    safe: Optional[bool]=None
    scheduler: Optional[SCHEDULER_LITERAL]=None
    scheduler_beta_start: Optional[float]=None
    scheduler_beta_end: Optional[float]=None
    scheduler_beta_schedule: Optional[str]=None
    # Custom model args
    model_prompt: Optional[str]=None
    model_prompt_2: Optional[str]=None
    model_negative_prompt: Optional[str]=None
    model_negative_prompt_2: Optional[str]=None
    # Invocation
    prompts: Optional[List[PromptDict]]=None
    prompt: Optional[str]=None
    prompt_2: Optional[str]=None
    negative_prompt: Optional[str]=None
    negative_prompt_2: Optional[str]=None
    clip_skip: Optional[int]=None
    tiling_unet: bool=False
    tiling_vae: bool=False
    tiling_size: Optional[int]=None
    tiling_stride: Optional[int]=None
    tiling_mask_type: Optional[MASK_TYPE_LITERAL]=None
    tiling_mask_kwargs: Optional[Dict[str, Any]]=None
    # Layers
    layers: List[Dict[str, Any]]=field(default_factory=list) #TODO: stronger type
    # Generation
    samples: int=1
    iterations: int=1
    seed: Optional[int]=None
    tile: Union[bool, Tuple[bool, bool], List[bool]]=False
    # Tweaks
    freeu_factors: Optional[Tuple[float, float, float, float]]=None
    guidance_scale: Optional[float]=None
    num_inference_steps: Optional[int]=None
    noise_offset: Optional[float]=None
    noise_method: NOISE_METHOD_LITERAL="perlin"
    noise_blend_method: LATENT_BLEND_METHOD_LITERAL="inject"
    # Animation
    animation_engine: Optional[ANIMATION_ENGINE_LITERAL]=None
    animation_frames: Optional[int]=None
    animation_rate: int=8
    frame_window_size: Optional[int]=16
    frame_window_stride: Optional[int]=4
    frame_decode_chunk_size: Optional[int]=None
    loop: bool=False
    motion_module: Optional[str]=None
    motion_scale: Optional[float]=None
    position_encoding_truncate_length: Optional[int]=None
    position_encoding_scale_length: Optional[int]=None
    num_denoising_iterations: Optional[int]=None
    # stable video
    svd_model: Optional[Literal["svd", "svd_xt"]]=None
    motion_bucket_id: int=127
    fps: int=7
    noise_aug_strength: float=0.02
    min_guidance_scale: float=1.0
    max_guidance_scale: float=3.0
    # dragnuwa
    motion_vectors: Optional[List[List[MotionVectorPointDict]]]=None
    optical_flow: Optional[Dict[OPTICAL_FLOW_METHOD_LITERAL, Union[List[Image],List[List[Image]]]]]=None
    motion_vector_repeat_window: bool=False
    gaussian_sigma: int=20
    # img2img
    strength: Optional[float]=None
    # Inpainting
    mask: Optional[Union[str, Image, List[Image]]]=None
    crop_inpaint: bool=True
    scale_inpaint: bool=True
    inpaint_feather: int=32
    inpaint_upscale_amount: float=0.5
    outpaint: bool=True
    outpaint_dilate: int=4
    # Refining
    refiner_start: Optional[float]=None
    refiner_strength: Optional[float]=None
    refiner_guidance_scale: float=DEFAULT_REFINER_GUIDANCE_SCALE
    refiner_aesthetic_score: float=DEFAULT_AESTHETIC_SCORE
    refiner_negative_aesthetic_score: float=DEFAULT_NEGATIVE_AESTHETIC_SCORE
    refiner_prompt: Optional[str]=None
    refiner_prompt_2: Optional[str]=None
    refiner_negative_prompt: Optional[str]=None
    refiner_negative_prompt_2: Optional[str]=None
    # Flags
    build_tensorrt: bool=False
    # Post-processing
    detailer_face_restore: bool=False
    detailer_face_inpaint: bool=False
    detailer_hand_inpaint: bool=False
    detailer_guidance_scale: Optional[float]=None
    detailer_inference_steps: Optional[int]=None
    detailer_inpaint_strength: float=0.25
    detailer_inpaint_dilate_ratio: float=0.33
    detailer_inpaint_blur: int=16
    detailer_inpaint_feather: int=16
    detailer_inpaint_min_size: int=64
    detailer_inpaint_ip_adapter_scale=0.65
    detailer_controlnet: Optional[CONTROLNET_LITERAL]=None
    detailer_controlnet_scale: float=1.0
    detailer_switch_pipeline: bool=False
    detailer_upscale_amount: float=0.5
    upscale: Optional[Union[UpscaleStepDict, List[UpscaleStepDict]]]=None
    interpolate_frames: Optional[int]=None
    reflect: bool=False

    @property
    def face_isolate_dilate(self) -> int:
        """
        The number of pixels to dilate face masks by
        """
        return 256

    @staticmethod
    def merge_prompts(*args: Tuple[Optional[str], float]) -> Optional[str]:
        """
        Merges prompts if they are not null
        """
        if all([not prompt for prompt, weight in args]):
            return None
        return "".join([
            f"({prompt}){weight}"
            for prompt, weight in args
            if prompt
        ])

    @staticmethod
    def parse_frame_range(frames: Optional[Union[int, List[int], str]], one_index: bool = False) -> Optional[List[int]]:
        """
        Parses a comma-separated frame string
        """
        offset = 1 if one_index else 0
        if isinstance(frames, int):
            return [frames-offset]
        elif isinstance(frames, list):
            frames = [int(f)-offset for f in frames]
            frames.sort()
            return frames
        elif isinstance(frames, str) and len(frames) > 0:
            frame_parts = frames.split(",")
            frame_indexes: List[int] = []
            for frame_part in frame_parts:
                frame_range = frame_part.split("-")
                if len(frame_range) == 1:
                    frame_indexes.append(int(frame_range[0])-offset)
                else:
                    frame_indexes.extend(list(range(int(frame_range[0])-offset, int(frame_range[1]))))
            return list(sorted(set(frame_indexes)))
        return None

    @classmethod
    def get_image_bounding_box(
        cls,
        mask: Image,
        size: int,
        feather: int=32
    ) -> List[Tuple[int, int]]:
        """
        Gets the feathered bounding box for an image
        """
        width, height = mask.size
        bbox = mask.getbbox()
        if bbox is None:
            return [(0, 0), (width, height)]

        x0, y0, x1, y1 = bbox

        # Add feather
        x0 = max(0, x0 - feather)
        x1 = min(width, x1 + feather)
        y0 = max(0, y0 - feather)
        y1 = min(height, y1 + feather)
        
        # Create centered frame about the bounding box
        bbox_width = x1 - x0
        bbox_height = y1 - y0

        if bbox_width < size:
            x0 = max(0, x0 - ((size - bbox_width) // 2))
            x1 = min(width, x0 + size)
            x0 = max(0, x1 - size)
        if bbox_height < size:
            y0 = max(0, y0 - ((size - bbox_height) // 2))
            y1 = min(height, y0 + size)
            y0 = max(0, y1 - size)

        return [(x0, y0), (x1, y1)]

    @classmethod
    def get_inpaint_bounding_box(
        cls,
        mask: Union[Image, List[Image]],
        size: int,
        feather: int=32
    ) -> List[Tuple[int, int]]:
        """
        Gets the bounding box of places inpainted
        """
        # Find bounding box
        (x0, y0) = (0, 0)
        (x1, y1) = (0, 0)
        for mask_image in (mask if isinstance(mask, list) else [mask]):
            (frame_x0, frame_y0), (frame_x1, frame_y1) = cls.get_image_bounding_box(
                mask=mask_image,
                size=size,
                feather=feather,
            )
            x0 = max(x0, frame_x0)
            y0 = max(y0, frame_y0)
            x1 = max(x1, frame_x1)
            y1 = max(y1, frame_y1)

        return [(x0, y0), (x1, y1)]

    @classmethod
    def paste_inpaint_image(
        cls,
        background: Image,
        foreground: Image,
        position: Tuple[int, int],
        inpaint_feather: int=32,
    ) -> Image:
        """
        Pastes the inpaint image on the background with an appropriately feathered mask.
        """
        from PIL import Image
        image = background.copy()

        width, height = image.size
        foreground_width, foreground_height = foreground.size
        left, top = position[:2]
        right, bottom = left + foreground_width, top + foreground_height

        feather_left = left > 0
        feather_top = top > 0
        feather_right = right < width
        feather_bottom = bottom < height

        mask = Image.new("L", (foreground_width, foreground_height), 255)
        for i in range(inpaint_feather):
            multiplier = (i + 1) / (inpaint_feather + 1)
            pixels = []
            if feather_left:
                pixels.extend([(i, j) for j in range(foreground_height)])
            if feather_top:
                pixels.extend([(j, i) for j in range(foreground_width)])
            if feather_right:
                pixels.extend([(foreground_width - i - 1, j) for j in range(foreground_height)])
            if feather_bottom:
                pixels.extend([(j, foreground_height - i - 1) for j in range(foreground_width)])
            for x, y in pixels:
                mask.putpixel((x, y), int(mask.getpixel((x, y)) * multiplier))

        image.paste(foreground, position[:2], mask=mask)
        return image

    @classmethod
    def find_overlap(
        cls,
        bbox1: Tuple[Tuple[int, int], Tuple[int, int]],
        bbox2: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> Optional[Tuple[int, int], Tuple[int, int]]:
        """
        Find the overlapping bounding box of two given bounding boxes.
        """
        (x00, y00), (x01, y01) = bbox1
        (x10, y10), (x11, y11) = bbox2

        ox0 = max(x00, x10)
        oy0 = max(y00, y10)
        ox1 = min(x01, x11)
        oy1 = min(y01, y11)

        if ox0 < ox1 and oy0 < oy1:
            return ((ox0, oy0), (ox1, oy1))
        else:
            return None

    @classmethod
    def distance(
        cls,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate the Euclidean distance between the centers of two bounding boxes
        """
        from math import sqrt
        (x00, y00), (x01, y01) = bbox1
        (x10, y10), (x11, y11) = bbox2
        cx1, cy1 = ((x00 + x01) / 2, (y00 + y01) / 2)
        cx2, cy2 = ((x10 + x11) / 2, (y10 + y11) / 2)
        return sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    @classmethod
    def group_bounding_boxes(
        cls,
        width: int,
        height: int,
        frames: List[List[Tuple[int, int, int, int]]]
    ) -> List[List[Optional[Tuple[int, Tuple[int, int, int, int]]]]]:
        """
        Groups bounding boxes by minimum distance (tracks individual features)
        """
        max_movement = 0.1 * min(width, height) # 10% of the smallest dimension
        groups = []

        def recurse_frames(frames: List[List[Tuple[int, int, int, int]]]) -> None:
            """
            Recurses through the frame array and tracks
            """
            frame = frames[0]
            unfound = []
            groups_found = []
            for i, bounding_box in enumerate(frame):
                nearest_group = None
                nearest_distance = None
                for j, group_bboxes in enumerate(groups):
                    if j in groups_found:
                        continue
                    last_known_position = [bbox[1] for bbox in group_bboxes if bbox][-1]
                    this_distance = cls.distance(bounding_box, last_known_position)
                    if nearest_group is None or nearest_distance is None or this_distance < nearest_distance:
                        nearest_group = j
                        nearest_distance = this_distance
                if nearest_distance is not None and nearest_distance < max_movement:
                    groups[nearest_group].append((i, bounding_box))
                    groups_found.append(nearest_group)
                else:
                    unfound.append((i, bounding_box))
            for j in range(len(groups)):
                if j not in groups_found:
                    groups[j].append(None)
            for i, bounding_box in unfound:
                groups.append(([None] * (0 if not groups else len(groups[0])-1)) + [(i, bounding_box)])
            if len(frames) > 1:
                recurse_frames(frames[1:])

        recurse_frames(frames)
        return groups

    @classmethod
    def extend_mask(cls, image: Image, move_up_pixels: int) -> Image:
        """
        Extends a mask vertically from its widest point
        """
        import cv2
        import numpy as np
        from PIL import Image

        # Convert image to grayscale and threshold to ensure we have a binary image
        gray = np.array(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find the horizontal projection by summing up the values of pixels along rows
        horizontal_projection = np.sum(binary, axis=1)

        # The widest point is where the horizontal projection will have the maximum value
        widest_row = np.argmax(horizontal_projection)
        # Get the left/right position
        indices = np.where(binary[widest_row])[0]
        left, right = min(indices), max(indices)

        # Split the image at the widest row
        top_part = binary[:widest_row, :]
        bottom_part = binary[widest_row:, :]

        # Create a new image that will contain the adjusted mask
        adjusted_image = np.zeros(tuple(binary.shape[:2]), dtype=np.uint8)

        # Fill the new image: place the top part higher by move_up_pixels
        adjusted_image[:widest_row-move_up_pixels, :] = top_part[move_up_pixels:]
        adjusted_image[widest_row-move_up_pixels:widest_row, left:right+1] = 255

        # Fill in the rest of the mask below the widest part
        adjusted_image[widest_row:, :] = bottom_part

        return Image.fromarray(adjusted_image)


    @property
    def upscale_steps(self) -> Iterator[UpscaleStepDict]:
        """
        Iterates over upscale steps.
        """
        if self.upscale is not None:
            if isinstance(self.upscale, list):
                for step in self.upscale:
                    yield step
            else:
                yield self.upscale

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns the keyword arguments that will passed to the pipeline invocation.
        """
        return {
            "width": self.width,
            "height": self.height,
            "strength": self.strength,
            "animation_frames": self.animation_frames,
            "tile": tuple(self.tile[:2]) if isinstance(self.tile, list) else self.tile,
            "freeu_factors": self.freeu_factors,
            "num_inference_steps": self.num_inference_steps,
            "num_results_per_prompt": self.samples,
            "noise_offset": self.noise_offset,
            "noise_method": self.noise_method,
            "noise_blend_method": self.noise_blend_method,
            "loop": self.loop,
            "tiling_vae": self.tiling_vae,
            "tiling_unet": self.tiling_unet,
            "tiling_size": self.tiling_size,
            "tiling_stride": self.tiling_stride,
            "tiling_mask_type": self.tiling_mask_type,
            "motion_scale": self.motion_scale,
            "frame_window_size": self.frame_window_size,
            "frame_window_stride": self.frame_window_stride,
            "frame_decode_chunk_size": self.frame_decode_chunk_size,
            "num_denoising_iterations": self.num_denoising_iterations,
            "guidance_scale": self.guidance_scale,
            "refiner_start": self.refiner_start,
            "refiner_strength": self.refiner_strength,
            "refiner_guidance_scale": self.refiner_guidance_scale,
            "refiner_aesthetic_score": self.refiner_aesthetic_score,
            "refiner_negative_aesthetic_score": self.refiner_negative_aesthetic_score,
            "refiner_prompt": self.refiner_prompt,
            "refiner_prompt_2": self.refiner_prompt_2,
            "refiner_negative_prompt": self.refiner_negative_prompt,
            "refiner_negative_prompt_2": self.refiner_negative_prompt_2,
            "ip_adapter_model": self.ip_adapter_model,
            "clip_skip": self.clip_skip
        }

    def remove_alpha(
        self,
        image: Image,
        pipeline: Optional[DiffusionPipelineManager]=None,
        mode: Optional[Literal["none", "noise", "shuffle"]]="noise",
        noise_method: Optional[NOISE_METHOD_LITERAL]="default",
        contrast: float=0.8
    ) -> Image:
        """
        Replaces the alpha channel in an image with noise.
        Uses the generator from the pipeline, if one exists.
        """
        if pipeline is not None and not pipeline.inpainter_is_sdxl:
            mode = None

        width, height = image.size

        if mode is None or mode == "none":
            # Strip alpha (fill with black)
            return image.convert("RGB")
        elif mode == "noise":
            # Fill alpha with noise
            from enfugue.diffusion.util import make_noise, tensor_to_image
            from PIL import Image, ImageStat
            # Get stats for image
            stat = ImageStat.Stat(image)
            extrema = list(image.getextrema())
            # Build noise channel-by-channel
            channels: List[Image] = []
            for i in range(3):
                # Find range for this channel in source image
                (c_min, c_max) = extrema[i]
                noise = make_noise(
                    method=noise_method, # type: ignore[arg-type]
                    width=width,
                    height=height,
                    channels=1,
                    batch_size=1,
                    generator=None if pipeline is None else pipeline.noise_generator,
                )
                # Scale noise based on image range
                noise = ((
                    (c_min/255.0 * noise) +
                    ((c_max-c_min)/255.0 * noise)
                ))
                # Scale noise based on passed contrast
                noise = (noise * contrast) + (1.0 - contrast)
                # Scale noise based on image mean
                noise *= stat.mean[i] / 255.0
                # Add to channel list
                channels.append(tensor_to_image(noise).convert("L"))
            # Merge channels
            noise_image = Image.merge("RGB", tuple(channels))
            noise_image.paste(image, mask=dilate_erode(image.split()[-1], -self.outpaint_dilate))
            return noise_image
        return image

    @classmethod
    def prepare_image(
        cls,
        width: int,
        height: int,
        image: Union[str, Image, List[Image]],
        animation_frames: Optional[int]=None,
        frames: Optional[List[int]]=None,
        fit: Optional[IMAGE_FIT_LITERAL]=None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL]=None,
        offset_x: Optional[int]=None,
        offset_y: Optional[int]=None,
        x: Optional[int]=None,
        y: Optional[int]=None,
        w: Optional[int]=None,
        h: Optional[int]=None,
        return_mask: bool=True,
    ) -> Union[Image, List[Image], Tuple[Image, Image], Tuple[List[Image], List[Image]]]:
        """
        Fits an image on the canvas and returns it and it's alpha mask
        """
        from PIL import Image

        if isinstance(image, str):
            image = get_frames_or_image_from_file(image)

        if w is not None and h is not None:
            fitted_image = fit_image(
                image=image,
                width=w,
                height=h,
                fit=fit,
                anchor=anchor,
                offset_left=offset_x,
                offset_top=offset_y
            )
        else:
            fitted_image = fit_image(
                image=image,
                width=width,
                height=height,
                fit=fit,
                anchor=anchor,
                offset_left=offset_x,
                offset_top=offset_y
            )

        if x is not None and y is not None:
            if isinstance(fitted_image, list):
                for i, img in enumerate(fitted_image):
                    blank_image = Image.new("RGBA", (width, height), (0,0,0,0))
                    if img.mode == "RGBA":
                        blank_image.paste(img, (x, y), img)
                    else:
                        blank_image.paste(img, (x, y))
                    fitted_image[i] = blank_image
            else:
                blank_image = Image.new("RGBA", (width, height), (0,0,0,0))
                if fitted_image.mode == "RGBA":
                    blank_image.paste(fitted_image, (x, y), fitted_image)
                else:
                    blank_image.paste(fitted_image, (x, y))
                fitted_image = blank_image

        if isinstance(fitted_image, list):
            if not animation_frames:
                fitted_image = fitted_image[0]
            else:
                requested_frames = animation_frames
                if frames is not None:
                    requested_frames = len(frames)
                fitted_image = fitted_image[:requested_frames]

        if not return_mask:
            return fitted_image

        if isinstance(fitted_image, list):
            image_mask = [
                Image.new("1", (width, height), (1))
                for i in range(len(fitted_image))
            ]
        else:
            image_mask = Image.new("1", (width, height), (1))

        black = Image.new("1", (width, height), (0))

        if isinstance(fitted_image, list):
            for i, img in enumerate(fitted_image):
                fitted_alpha = img.split()[-1]
                fitted_alpha_inverse_clamp = Image.eval(fitted_alpha, lambda a: 0 if a > 128 else 255)
                image_mask[i].paste(black, mask=fitted_alpha_inverse_clamp)
        else:
            fitted_alpha = fitted_image.split()[-1]
            fitted_alpha_inverse_clamp = Image.eval(fitted_alpha, lambda a: 0 if a > 128 else 255)
            image_mask.paste(black, mask=fitted_alpha_inverse_clamp) # type: ignore[attr-defined]

        return fitted_image, image_mask

    @classmethod
    def assemble(
        cls,
        size: int=512,
        image: Optional[Union[str, Image, List[Image], ImageDict]]=None,
        ip_adapter_images: Optional[List[IPAdapterImageDict]]=None,
        control_images: Optional[List[ControlImageDict]]=None,
        loop: Union[bool, str]=False,
        **kwargs: Any
    ) -> LayeredInvocation:
        """
        Assembles an invocation from layers, standardizing arguments
        """
        invocation_kwargs = dict([
            (k, v) for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        ])
        ignored_kwargs = set(list(kwargs.keys())) - set(list(invocation_kwargs.keys()))

        # Add directly passed images to layers
        added_layers = []
        if image:
            if isinstance(image, dict):
                added_layers.append(image)
            else:
                added_layers.append({"image": image})
        if ip_adapter_images:
            for ip_adapter_image in ip_adapter_images:
                added_layers.append({
                    "image": ip_adapter_image["image"],
                    "ip_adapter_scale": ip_adapter_image.get("scale", 1.0),
                    "face_only": ip_adapter_image.get("face_only", False),
                    "fit": ip_adapter_image.get("fit", None),
                    "anchor": ip_adapter_image.get("anchor", None),
                    "frame": ip_adapter_image.get("frame", None),
                })
        if control_images:
            for control_image in control_images:
                added_layers.append({
                    "image": control_image["image"],
                    "fit": control_image.get("fit", None),
                    "anchor": control_image.get("anchor", None),
                    "frame": control_image.get("frame", None),
                    "control_units": [
                        {
                            "controlnet": control_image["controlnet"],
                            "scale": control_image.get("scale", 1.0),
                            "start": control_image.get("start", None),
                            "end": control_image.get("end", None),
                            "process": control_image.get("process", True),
                        }
                    ]
                })

        # Reassign layers
        if "layers" in invocation_kwargs:
            invocation_kwargs["layers"].extend(added_layers)
        else:
            invocation_kwargs["layers"] = added_layers

        # Gather size of images for defaults and trim video
        animation_frames = invocation_kwargs.get("animation_frames", None)
        image_width, image_height = 0, 0
        for layer in invocation_kwargs["layers"]:
            # Standardize images
            if isinstance(layer["image"], str):
                layer["image"] = get_frames_or_image_from_file(layer["image"])
            elif not isinstance(layer["image"], list):
                layer["image"] = get_frames_or_image(layer["image"])

            skip_frames = layer.pop("skip_frames", None)
            divide_frames = layer.pop("divide_frames", None)
            frames = cls.parse_frame_range(layer.pop("frame", None), True)

            if skip_frames and isinstance(layer["image"], list):
                layer["image"] = layer["image"][skip_frames:]

            if divide_frames and isinstance(layer["image"], list):
                layer["image"] = [
                    img for i, img in enumerate(layer["image"])
                    if i % divide_frames == 0
                ]

            if animation_frames and frames is not None:
                layer["frame"] = frames


            if isinstance(layer["image"], list):
                # Minimize the number of images we pass
                if animation_frames:
                    layer_frames = animation_frames
                    if frames is not None:
                        layer_frames = len(frames)
                    layer["image"] = layer["image"][:layer_frames]
                else:
                    layer["image"] = layer["image"][0]

            # Check if this image is visible
            if (
                layer.get("image", None) is not None and
                layer.get("visibility", None) in ["visible", "denoised"]
            ):
                layer_x = layer.get("x", 0)
                layer_y = layer.get("y", 0)

                if isinstance(layer["image"], list):
                    image_w, image_h = layer["image"][0].size
                else:
                    image_w, image_h = layer["image"].size

                layer_w = layer.get("w", image_w)
                layer_h = layer.get("h", image_h)
                image_width = max(image_width, layer_x + layer_w)
                image_height = max(image_height, layer_y + layer_h)

        # Make sure a size is set
        if not invocation_kwargs.get("width", None):
            invocation_kwargs["width"] = image_width if image_width else size
        if not invocation_kwargs.get("height", None):
            invocation_kwargs["height"] = image_height if image_height else size

        # Make sure size is properly divisible
        divisor = 8 if not invocation_kwargs.get("animation_engine", None) == "svd" else 64
        floor_width = floor(invocation_kwargs["width"] / divisor) * divisor
        floor_height = floor(invocation_kwargs["height"] / divisor) * divisor
        if invocation_kwargs["width"] != floor_width or invocation_kwargs["height"] != floor_height:
            logger.warning(f"Width and height must be divisible by {divisor}, cropping from {invocation_kwargs['width']}x{invocation_kwargs['height']} to {floor_width}x{floor_height}")
            invocation_kwargs["width"] = floor_width
            invocation_kwargs["height"] = floor_height

        # Add seed if not set
        if not invocation_kwargs.get("seed", None):
            invocation_kwargs["seed"] = randint(0,2**32)

        # Check loop
        if isinstance(loop, bool):
            invocation_kwargs["loop"] = loop
        elif isinstance(loop, str):
            invocation_kwargs["loop"] = loop == "loop"
            invocation_kwargs["reflect"] = loop == "reflect"

        # Check samples and iterations (don't do multiple if animating)
        if invocation_kwargs.get("animation_frames", None):
            invocation_kwargs["samples"] = 1
            invocation_kwargs["iterations"] = 1

        if ignored_kwargs:
            logger.warning(f"Ignored keyword arguments: {ignored_kwargs}")

        return cls(**invocation_kwargs)

    @classmethod
    def minimize_dict(
        cls,
        kwargs: Dict[str, Any],
        has_refiner: bool = False
    ) -> Dict[str, Any]:
        """
        Pops unnecessary variables from an invocation dict
        """
        all_keys = list(kwargs.keys())
        layers = kwargs.get("layers", None)
        if layers is None:
            layers = []

        minimal_keys = []

        has_noise = bool(kwargs.get("noise_offset", None))
        will_inpaint = bool(kwargs.get("mask", None))
        will_animate = bool(kwargs.get("animation_frames", None))

        has_ip_adapter = bool(kwargs.get("ip_adapter_images", None))
        has_ip_adapter = has_ip_adapter or any([
            bool(layer.get("ip_adapter_scale", None))
            for layer in layers
        ])

        for key in all_keys:
            value = kwargs[key]
            if value is None:
                continue
            if "layers" == key and not value:
                continue
            if "tile" == key and value == False:
                continue
            if "refiner" in key and not has_refiner:
                continue
            if "inpaint" in key and "detailer" not in key and not will_inpaint:
                continue
            if "ip_adapter" in key and not has_ip_adapter:
                continue
            if (
                (
                    "motion" in key or
                    "temporal" in key or
                    "animation" in key or
                    "frame" in key or
                    key in ["loop", "reflect", "num_denoising_iterations"]
                )
                and not will_animate
            ):
                continue

            if "noise" in key and not has_noise:
                continue

            minimal_keys.append(key)

        return dict([
            (key, kwargs[key])
            for key in minimal_keys
        ])

    def serialize(self) -> Dict[str, Any]:
        """
        Assembles self into a serializable dict
        """
        return self.minimize_dict(asdict(self))

    @contextmanager
    def preprocessors(
        self,
        pipeline: DiffusionPipelineManager
    ) -> Iterator[Dict[str, Callable[[Image], Image]]]:
        """
        Gets all preprocessors needed for this invocation
        """
        needs_pose_detector = False
        needs_background_remover = False
        needs_interpolator = False
        needs_control_processors = []
        visible_image_frames: List[bool] = [False] * (1 if not self.animation_frames else self.animation_frames)
        to_check: List[Dict[str, Any]] = []

        if self.layers is not None:
            for layer in self.layers:
                if layer.get("image", None) is not None:
                    to_check.append(layer)

        for image_dict in to_check:
            if image_dict.get("remove_background", False):
                needs_background_remover = True
            if image_dict.get("face_only", False):
                needs_pose_detector = True
            if image_dict.get("visibility", None) in ["denoised", "visible"]:
                layer_frames = 1 if not isinstance(image_dict["image"], list) else len(image_dict["image"])
                layer_frame_range = self.parse_frame_range(image_dict.get("frame", None))

                if layer_frame_range is None:
                    visible_image_frames[:layer_frames] = [True] * layer_frames
                else:
                    for layer_frame in layer_frame_range:
                        visible_image_frames[layer_frame] = True
            for control_dict in image_dict.get("control_units", []):
                if control_dict.get("process", True) and control_dict.get("controlnet", None) is not None:
                    if control_dict["controlnet"] == "pose":
                        needs_pose_detector = True
                    else:
                        needs_control_processors.append(control_dict["controlnet"])

        if len(visible_image_frames) > 1:
            has_visible_frame = False
            is_visible = False
            for frame_is_visible in visible_image_frames:
                if has_visible_frame and not is_visible and frame_is_visible:
                    needs_interpolator = True
                has_visible_frame = has_visible_frame or frame_is_visible
                is_visible = frame_is_visible

        with ExitStack() as stack:
            processors: Dict[str, Callable[[Image], Image]] = {}
            if needs_background_remover:
                processors["background_remover"] = stack.enter_context(
                    pipeline.background_remover.remover()
                )
            if needs_interpolator:
                processors["interpolator"] = stack.enter_context( # type: ignore[assignment]
                    pipeline.interpolator.film()
                )
            if needs_pose_detector:
                processors["pose"] = stack.enter_context(
                    pipeline.control_image_processor.pose_detector.best()
                )
            if needs_control_processors:
                processor_names = list(set(needs_control_processors))
                with pipeline.control_image_processor.processors(*processor_names) as processor_callables:
                    processors = {**processors, **dict(zip(processor_names, processor_callables))}
                    yield processors
            else:
                yield processors

    def preprocess(
        self,
        pipeline: DiffusionPipelineManager,
        intermediate_dir: Optional[str]=None,
        raise_when_unused: bool=True,
        task_callback: Optional[Callable[[str], None]]=None,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Processes/transforms arguments
        """
        from PIL import Image, ImageOps
        from enfugue.diffusion.util.prompt_util import Prompt

        # Gather images for preprocessing
        control_images: Dict[str, List[Dict]] = {}
        ip_adapter_images = []
        invocation_mask = None
        invocation_image = None
        no_inference = False
        animation_frames = None if not self.animation_frames or self.animation_engine == "svd" else self.animation_frames

        if self.layers:
            if task_callback is not None:
                task_callback("Pre-processing layers")

            # Blank images used for merging
            black = Image.new("1", (self.width, self.height), (0))
            white = Image.new("1", (self.width, self.height), (1))

            mask = self.mask

            # Standardize mask
            if mask is not None:
                if isinstance(mask, dict):
                    invert = mask.get("invert", False)
                    mask = mask.get("image", None)
                    if isinstance(mask, str):
                        mask = get_frames_or_image_from_file(mask)
                    if not mask:
                        raise BadRequestError("Expected mask dictionary to have 'image' key")
                    if invert:
                        from PIL import ImageOps
                        if isinstance(mask, list):
                            mask = [
                                ImageOps.invert(img) for img in mask
                            ]
                        else:
                            mask = ImageOps.invert(mask)
                elif isinstance(mask, str):
                    mask = get_frames_or_image_from_file(mask)

                mask = self.prepare_image(
                    width=self.width,
                    height=self.height,
                    image=mask,
                    animation_frames=animation_frames,
                    return_mask=False
                )

            if self.animation_frames:
                invocation_mask = [
                    white.copy()
                    for i in range(self.animation_frames)
                ]
                invocation_image = [
                    Image.new("RGBA", (self.width, self.height), (0,0,0,0))
                    for i in range(self.animation_frames)
                ]
            else:
                invocation_image = Image.new("RGBA", (self.width, self.height), (0,0,0,0))
                invocation_mask = white.copy()

            has_invocation_image = False

            # Get a count of preprocesses required
            images_to_preprocess = 0
            visible_frames: List[bool] = [False] * (1 if not self.animation_frames else self.animation_frames)
            for i, layer in enumerate(self.layers):
                layer_image = layer.get("image", None)
                layer_frames = self.parse_frame_range(layer.get("frame", None), False)

                if isinstance(layer_image, str):
                    layer_image = get_frames_or_image_from_file(layer_image)
                    layer["image"] = layer_image

                image_count = len(layer_image) if isinstance(layer_image, list) else 1
                if image_count == 1 and layer_frames is not None:
                    image_count = len(layer_frames)
                elif image_count > 1 and self.animation_frames:
                    if layer_frames is not None:
                        layer_frame = range(layer_frames[0], min(layer_frames[0] + image_count, self.animation_frames))
                        image_count = len(layer_frames)
                    else:
                        image_count = min(image_count, self.animation_frames)
                        layer_frames = list(range(image_count))

                if layer_frames is None:
                    if self.animation_frames:
                        layer_frames = list(range(self.animation_frames))
                    else:
                        layer_frames = [0]

                if layer.get("remove_background", False):
                    images_to_preprocess += image_count

                if layer.get("face_only", False):
                    images_to_preprocess += image_count

                if layer.get("visibility", None) in ["visible", "denoised"]:
                    for frame in layer_frames:
                        visible_frames[frame] = True
                control_units = layer.get("control_units", None)
                if control_units:
                    for control_unit in control_units:
                        if control_unit.get("process", True):
                            images_to_preprocess += image_count

            # Get interpolated frames
            needs_interpolation = [False] * (1 if not self.animation_frames else self.animation_frames)
            last_visible_frame: Optional[int] = None
            for i, frame_is_visible in enumerate(visible_frames):
                if frame_is_visible:
                    if last_visible_frame is not None:
                        interpolated_frames = i - last_visible_frame - 1
                        if interpolated_frames > 0:
                            needs_interpolation[last_visible_frame+1:i] = [True] * interpolated_frames
                    last_visible_frame = i

            images_to_preprocess += sum(needs_interpolation)
            images_preprocessed = 0
            last_frame_time = datetime.now()
            frame_times = []

            def trigger_preprocess_callback(image: Image) -> Image:
                """
                Triggers the preprocessor callback
                """
                nonlocal last_frame_time
                nonlocal images_preprocessed
                if progress_callback is not None:
                    images_preprocessed += 1
                    frame_time = datetime.now()
                    frame_seconds = (frame_time - last_frame_time).total_seconds()
                    frame_times.append(frame_seconds)
                    frame_time_samples = min(len(frame_times), 8)
                    frame_time_average = sum(frame_times[-8:]) / frame_time_samples

                    progress_callback(images_preprocessed, images_to_preprocess, 1 / frame_time_average)

                    last_frame_time = frame_time
                return image

            # Preprocess images
            if images_to_preprocess:
                logger.debug(f"Pre-processing layers, with {images_to_preprocess} image processing step(s)")

            with self.preprocessors(pipeline) as processors:
                # Iterate through layers
                for i, layer in enumerate(self.layers):
                    # Basic information for layer
                    w = layer.get("w", None)
                    h = layer.get("h", None)
                    x = layer.get("x", None)
                    y = layer.get("y", None)

                    layer_image = layer.get("image", None)
                    fit = layer.get("fit", None)
                    anchor = layer.get("anchor", None)
                    offset_x = layer.get("offset_x", None)
                    offset_y = layer.get("offset_y", None)
                    opacity = layer.get("opacity", None)
                    remove_background = layer.get("remove_background", None)
                    frames = self.parse_frame_range(layer.get("frame", None), False)

                    # Capabilities of layer
                    visibility = layer.get("visibility", None)
                    denoise = visibility == "denoised"

                    passthrough = visibility == "visible"

                    prompt_scale = layer.get("ip_adapter_scale", False)
                    control_units = layer.get("control_units", [])

                    if not layer_image:
                        logger.warning(f"No image, skipping laying {i}")
                        continue

                    if isinstance(layer_image, str):
                        layer_image = get_frames_or_image_from_file(layer_image)

                    if remove_background:
                        if isinstance(layer_image, list):
                            layer_image = [
                                trigger_preprocess_callback(processors["background_remover"](img))
                                for i, img in enumerate(layer_image)
                                if i < ((1 if not self.animation_frames else self.animation_frames) - frame)
                            ]
                        else:
                            layer_image = trigger_preprocess_callback(processors["background_remover"](layer_image))

                    fit_layer_image, fit_layer_mask = self.prepare_image(
                        width=self.width,
                        height=self.height,
                        image=layer_image,
                        fit=fit,
                        anchor=anchor,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        animation_frames=self.animation_frames,
                        frames=frames,
                        w=w,
                        h=h,
                        x=x,
                        y=y
                    )

                    if isinstance(fit_layer_mask, list):
                        inverse_fit_layer_mask = [
                            ImageOps.invert(img)
                            for img in fit_layer_mask
                        ]
                    else:
                        inverse_fit_layer_mask = ImageOps.invert(fit_layer_mask)

                    if denoise or passthrough:
                        has_invocation_image = True
                        image_paste_mask = fit_layer_mask

                        if opacity is not None:
                            if isinstance(image_paste_mask, list):
                                image_paste_mask = [
                                    img.convert("L")
                                    for img in image_paste_mask
                                ]
                                image_paste_mask = [
                                    Image.eval(img, lambda a: min(a, int(opacity * 255)))
                                    for img in image_paste_mask
                                ]
                            else:
                                image_paste_mask = image_paste_mask.convert("L")
                                image_paste_mask = Image.eval(image_paste_mask, lambda a: min(a, int(opacity * 255)))

                        if isinstance(fit_layer_image, list):
                            fit_layer_images = len(fit_layer_image)
                            if frames is None:
                                if isinstance(invocation_image, list):
                                    frames = list(range(len(invocation_image)))
                                else:
                                    frames = [0]
                            for i, frame in enumerate(frames):
                                index = i if i < fit_layer_images else fit_layer_images - 1
                                invocation_image[frame].paste( # type: ignore[index]
                                    fit_layer_image[index],
                                    mask=fit_layer_mask[index]
                                )
                                if passthrough:
                                    invocation_mask[frame].paste( # type: ignore[index]
                                        black,
                                        mask=fit_layer_mask[index]
                                    )
                        elif isinstance(invocation_image, list):
                            if frames is None:
                                frames = list(range(len(invocation_image)))
                            for frame in frames:
                                invocation_image[frame].paste(fit_layer_image, mask=image_paste_mask)
                                if passthrough:
                                    invocation_mask[frame].paste(black, mask=fit_layer_mask) # type: ignore[index]
                        else:
                            invocation_image.paste(fit_layer_image, mask=image_paste_mask) # type: ignore[attr-defined]
                            if passthrough:
                                invocation_mask.paste(black, mask=fit_layer_mask) # type: ignore[union-attr]

                    if prompt_scale:
                        # ip adapter
                        face_only = layer.get("face_only", False)
                        if face_only:
                            face_mask = processors["pose"].detail_mask(layer_image, include_face=True, include_hands=False) # type: ignore[attr-defined]
                            face_mask = dilate_erode(face_mask, self.face_isolate_dilate)
                            (x0, y0), (x1, y1) = self.get_inpaint_bounding_box(face_mask, size=512, feather=64)
                            ip_image = Image.new("RGB", (x1-x0, y1-y0), (0,0,0)) # Full black
                            ip_image.paste(layer_image, (-x0, -y0), mask=face_mask.convert("L"))
                        else:
                            ip_image = layer_image

                        ip_adapter_images.append({
                            "image": ip_image,
                            "scale": float(prompt_scale)
                        })

                    if control_units:
                        for control_unit in control_units:
                            controlnet = control_unit["controlnet"]

                            if controlnet not in control_images:
                                control_images[controlnet] = []

                            if control_unit.get("process", True):
                                if isinstance(fit_layer_image, list):
                                    control_image = [
                                        trigger_preprocess_callback(processors[controlnet](img))
                                        for img in fit_layer_image
                                    ]
                                else:
                                    control_image = trigger_preprocess_callback(processors[controlnet](fit_layer_image))
                            elif control_unit.get("invert", False):
                                if isinstance(fit_layer_image, list):
                                    control_image = [
                                        ImageOps.invert(img)
                                        for img in fit_layer_image
                                    ]
                                else:
                                    control_image = ImageOps.invert(fit_layer_image)
                            else:
                                control_image = fit_layer_image

                            control_images[controlnet].append({
                                "start": control_unit.get("start", 0.0),
                                "end": control_unit.get("end", 1.0),
                                "scale": control_unit.get("scale", 1.0),
                                "frame": frames,
                                "image": control_image,
                            })

                # Perform any interpolation and copy/paste
                if has_invocation_image and isinstance(invocation_image, list):
                    from enfugue.diffusion.util import interpolate_frames
                    visible_frame_indices = [i for i, is_visible in enumerate(visible_frames) if is_visible]
                    first_visible_frame = min(visible_frame_indices)
                    last_visible_frame = max(visible_frame_indices)
                    total_frames = len(invocation_image)

                    # Copy beginning
                    for i in range(first_visible_frame):
                        invocation_image[i].paste(invocation_image[first_visible_frame])

                    # Copy ending
                    for i in range(last_visible_frame+1, total_frames):
                        invocation_image[i].paste(invocation_image[last_visible_frame])

                    # Interpolate middles
                    def interpolator_progress_callback(step: int, total: int, rate: float) -> None:
                        if progress_callback is not None:
                            progress_callback(images_preprocessed + step, images_preprocessed + total, rate)

                    interpolate_start = None
                    for i, is_interpolated in enumerate(needs_interpolation):
                        if is_interpolated:
                            if interpolate_start is None:
                                interpolate_start = i
                        elif interpolate_start is not None:
                            multiplier = i - interpolate_start
                            logger.debug(f"Interpolating between frames {interpolate_start} and {i}.")
                            interpolated = list(interpolate_frames(
                                [invocation_image[interpolate_start-1], invocation_image[i]],
                                multiplier=multiplier,
                                interpolate=processors["interpolator"], # type: ignore[arg-type]
                                progress_callback=interpolator_progress_callback
                            ))
                            invocation_image[interpolate_start-1:i+1] = interpolated
                            images_preprocessed += multiplier
                            interpolate_start = None

            if not has_invocation_image:
                invocation_image = None
                invocation_mask = None
            else:
                if mask:
                    if isinstance(invocation_mask, list):
                        if not isinstance(mask, list):
                            mask = [mask.copy() for i in range(len(invocation_mask))] # type: ignore[union-attr]
                        for i in range(len(mask)):
                            mask[i].paste(
                                white,
                                mask=Image.eval(
                                    dilate_erode(invocation_mask[i], self.outpaint_dilate),
                                    lambda a: 0 if a < 128 else 255
                                )
                            )
                        invocation_mask = [
                            img.convert("L")
                            for img in mask
                        ]
                    else:
                        # Final mask merge
                        mask.paste( # type: ignore[union-attr]
                            white,
                            mask=Image.eval(
                                dilate_erode(invocation_mask, self.outpaint_dilate),
                                lambda a: 0 if a < 128 else 255
                            )
                        )
                        invocation_mask = mask.convert("L") # type: ignore[union-attr]
                else:
                    if isinstance(invocation_mask, list):
                        invocation_mask = [
                            dilate_erode(img, self.outpaint_dilate).convert("L") # type: ignore[union-attr]
                            for img in invocation_mask
                        ]
                    else:
                        invocation_mask = dilate_erode(invocation_mask, self.outpaint_dilate).convert("L") # type: ignore[union-attr]

                # Evaluate mask
                mask_max, mask_min = None, None
                if isinstance(invocation_mask, list):
                    for img in invocation_mask:
                        this_max, this_min = img.getextrema()
                        if mask_max is None:
                            mask_max = this_max
                        else:
                            mask_max = max(mask_max, this_max) # type: ignore[unreachable]
                        if mask_min is None:
                            mask_min = this_min
                        else:
                            mask_min = min(mask_min, this_min) # type: ignore[unreachable]
                else:
                    mask_max, mask_min = invocation_mask.getextrema()

                if mask_max == mask_min == 0:
                    # Nothing to do
                    if raise_when_unused:
                        raise BadRequestError("Nothing to do - canvas is covered by non-denoised images. Either modify the canvas such that there is blank space to be filled, enable denoising on an image on the canvas, or add inpainting.")
                    # Might have no invocation
                    invocation_mask = None
                    no_inference = True
                elif mask_max == mask_min == 255:
                    # No inpainting
                    invocation_mask = None
                elif not self.outpaint and not mask:
                    # Disabled outpainting
                    invocation_mask = None

        # Evaluate prompts
        prompts = self.prompts
        if prompts:
            # Prompt travel
            prompts = [
                Prompt( # type: ignore[misc]
                    positive=self.merge_prompts(
                        (prompt["positive"], 1.0),
                        (self.model_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    positive_2=self.merge_prompts(
                        (prompt.get("positive_2",None), 1.0),
                        (self.model_prompt_2, MODEL_PROMPT_WEIGHT)
                    ),
                    negative=self.merge_prompts(
                        (prompt.get("negative",None), 1.0),
                        (self.model_negative_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    negative_2=self.merge_prompts(
                        (prompt.get("negative_2",None), 1.0),
                        (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
                    ),
                    start=prompt.get("start",None),
                    end=prompt.get("end",None),
                    weight=prompt.get("weight", 1.0)
                )
                for prompt in prompts
            ]
        elif self.prompt is not None:
            prompts = [
                Prompt( # type: ignore[list-item]
                    positive=self.merge_prompts(
                        (self.prompt, 1.0),
                        (self.model_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    positive_2=self.merge_prompts(
                        (self.prompt_2, 1.0),
                        (self.model_prompt_2, MODEL_PROMPT_WEIGHT)
                    ),
                    negative=self.merge_prompts(
                        (self.negative_prompt, 1.0),
                        (self.model_negative_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    negative_2=self.merge_prompts(
                        (self.negative_prompt_2, 1.0),
                        (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
                    )
                )
            ]

        # Refiner prompts
        refiner_prompts = {
            "refiner_prompt": self.merge_prompts(
                (self.refiner_prompt, 1.0),
                (self.model_prompt, MODEL_PROMPT_WEIGHT)
            ),
            "refiner_prompt_2": self.merge_prompts(
                (self.refiner_prompt_2, 1.0),
                (self.model_prompt_2, MODEL_PROMPT_WEIGHT)
            ),
            "refiner_negative_prompt": self.merge_prompts(
                (self.refiner_negative_prompt, 1.0),
                (self.model_negative_prompt, MODEL_PROMPT_WEIGHT)
            ),
            "refiner_negative_prompt_2": self.merge_prompts(
                (self.refiner_negative_prompt_2, 1.0),
                (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
            )
        }

        # Completed pre-processing
        results_dict: Dict[str, Any] = {
            "no_inference": no_inference or (self.layers and (not invocation_mask and not self.strength and not control_images and not ip_adapter_images)),
            "animation_frames": animation_frames
        }

        if invocation_image:
            results_dict["image"] = invocation_image
            if invocation_mask:
                results_dict["mask"] = invocation_mask
        if control_images:
            results_dict["control_images"] = control_images
        if ip_adapter_images:
            results_dict["ip_adapter_images"] = ip_adapter_images
        if prompts:
            results_dict["prompts"] = prompts

        results_dict = self.minimize_dict(
            {
                **self.kwargs,
                **refiner_prompts,
                **results_dict
            },
            has_refiner=bool(self.refiner)
        )

        return results_dict

    def execute(
        self,
        pipeline: Optional[DiffusionPipelineManager] = None,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        This is the main interface for execution.

        The first step will be the one that executes with the selected number of samples,
        and then each subsequent step will be performed on the number of outputs from the
        first step.
        """
        if pipeline is None:
            from enfugue.diffusion.manager import DiffusionPipelineManager
            pipeline = DiffusionPipelineManager()
        # We import here so this file can be imported by processes without initializing torch
        from diffusers.utils.pil_utils import PIL_INTERPOLATION

        if task_callback is None:
            task_callback = lambda arg: None

        # Set up the pipeline
        safe_status = pipeline.safe # Store this in case we override
        pipeline.set_task_callback(task_callback)
        results = None
        try:
            original_image_callback: Optional[Callable] = None
            cropped_inpaint_position = None
            background = None
            has_post_processing = bool(self.upscale) or self.detailer_face_restore or ((self.detailer_face_inpaint or self.detailer_hand_inpaint) and self.detailer_inpaint_strength)

            if self.animation_frames:
                has_post_processing = has_post_processing or bool(self.interpolate_frames) or self.reflect or self.animation_engine == "svd"

            inference_image_callback = image_callback

            invocation_kwargs = self.preprocess(
                pipeline,
                raise_when_unused=not has_post_processing,
                task_callback=task_callback,
                progress_callback=progress_callback,
            )

            no_inference = invocation_kwargs.pop("no_inference", False)

            if no_inference:
                if "image" not in invocation_kwargs:
                    raise BadRequestError("No inference and no images.")
                else:
                    logger.debug("No inference necessary, skipping to post-processing.")
                images = invocation_kwargs["image"]
                if not isinstance(images, list):
                    images = [images]
                nsfw = [False] * len(images)
            else:
                # Check for a prompt
                if not invocation_kwargs.get("prompt", None) and (
                    not invocation_kwargs.get("prompts", None) or
                    all([
                        not prompt.positive for prompt in invocation_kwargs["prompts"]
                    ])
                ):
                    raise BadRequestError("A text prompt was required but not found. Please enter a prompt.")

                # Prepare the pipeline manager
                self.prepare_pipeline(pipeline)

                # Determine if we're doing cropped inpainting
                if invocation_kwargs.get("mask", None) is not None and self.crop_inpaint:
                    inpaint_size = self.tiling_size if self.tiling_size else 1024 if pipeline.inpainter_is_sdxl else 512

                    (x0, y0), (x1, y1) = self.get_inpaint_bounding_box(
                        invocation_kwargs["mask"],
                        size=inpaint_size if not self.scale_inpaint else 64,
                        feather=self.inpaint_feather
                    )

                    if isinstance(invocation_kwargs["mask"], list):
                        mask_width, mask_height = invocation_kwargs["mask"][0].size
                    else:
                        mask_width, mask_height = invocation_kwargs["mask"].size

                    bbox_width = x1 - x0
                    bbox_height = y1 - y0

                    pixel_ratio = (bbox_height * bbox_width) / (mask_width * mask_height)
                    pixel_savings = (1.0 - pixel_ratio) * 100

                    if pixel_ratio < 0.75:
                        logger.debug(f"Calculated pixel area savings of {pixel_savings:.1f}% by cropping to ({x0}, {y0}), ({x1}, {y1}) ({bbox_width}px by {bbox_height}px)")
                        cropped_inpaint_position = (x0, y0, x1, y1)
                    else:
                        logger.debug(
                            f"Calculated pixel area savings of {pixel_savings:.1f}% are insufficient, will not crop"
                        )

                if cropped_inpaint_position is not None:
                    # Get copies prior to crop
                    if isinstance(invocation_kwargs["image"], list):
                        background = [
                            img.copy()
                            for img in invocation_kwargs["image"]
                        ]
                    else:
                        background = invocation_kwargs["image"].copy()

                    # Get sizes
                    x0, y0, x1, y1 = cropped_inpaint_position
                    width = x1 - x0
                    height = y1 - y0
                    original_width = width
                    original_height = height

                    # First wrap callbacks if needed
                    if image_callback is not None:
                        # Hijack image callback to paste onto background
                        def pasted_image_callback(images: List[Image]) -> None:
                            """
                            Paste the images then callback.
                            """
                            if isinstance(background, list):
                                images = [
                                    self.paste_inpaint_image(
                                        (background[i] if i < len(background) else background[-1]),
                                        image.resize((original_width, original_height)),
                                        cropped_inpaint_position
                                    )
                                    for i, image in enumerate(images)
                                ]
                            else:
                                images = [
                                    self.paste_inpaint_image(
                                        background,
                                        image.resize((original_width, original_height)),
                                        cropped_inpaint_position
                                    )
                                    for image in images
                                ]

                            image_callback(images) # type: ignore

                        inference_image_callback = pasted_image_callback

                    # Now crop images
                    if isinstance(invocation_kwargs["image"], list):
                        invocation_kwargs["image"] = [
                            img.crop(cropped_inpaint_position)
                            for img in invocation_kwargs["image"]
                        ]
                        invocation_kwargs["mask"] = [
                            img.crop(cropped_inpaint_position)
                            for img in invocation_kwargs["mask"]
                        ]
                    else:
                        invocation_kwargs["image"] = invocation_kwargs["image"].crop(cropped_inpaint_position)
                        invocation_kwargs["mask"] = invocation_kwargs["mask"].crop(cropped_inpaint_position)

                    # Also crop control images
                    if "control_images" in invocation_kwargs:
                        for controlnet in invocation_kwargs["control_images"]:
                            for image_dict in invocation_kwargs["control_images"][controlnet]:
                                image_dict["image"] = image_dict["image"].crop(cropped_inpaint_position)

                    # Scale images if needed
                    if self.scale_inpaint and (inpaint_size > original_width or inpaint_size > original_height):
                        scale_size = inpaint_size / max(original_width, original_height) + self.inpaint_upscale_amount
                        logger.debug(f"Scaling inpaint by {scale_size}")
                        width = int(((width * scale_size) // 8) * 8)
                        height = int(((height * scale_size) // 8) * 8)
                        if isinstance(invocation_kwargs["image"], list):
                            invocation_kwargs["image"] = [
                                img.resize((width, height))
                                for img in invocation_kwargs["image"]
                            ]
                            invocation_kwargs["mask"] = [
                                img.resize((width, height), resample=PIL_INTERPOLATION["nearest"])
                                for img in invocation_kwargs["mask"]
                            ]
                        else:
                            invocation_kwargs["image"] = invocation_kwargs["image"].resize((width, height))
                            invocation_kwargs["mask"] = invocation_kwargs["mask"].resize((width, height), resample=PIL_INTERPOLATION["nearest"])

                    # Assign height and width
                    invocation_kwargs["width"] = width
                    invocation_kwargs["height"] = height

                # Execute primary inference
                results, nsfw = self.execute_inference(
                    pipeline,
                    task_callback=task_callback,
                    progress_callback=progress_callback,
                    image_callback=inference_image_callback,
                    image_callback_steps=image_callback_steps,
                    invocation_kwargs=invocation_kwargs
                )

            if background is not None and cropped_inpaint_position is not None:
                # Paste the image back onto the background
                x0, y0, x1, y1 = cropped_inpaint_position
                width = x1 - x0
                height = y1 - y0
                images = [
                    self.paste_inpaint_image(
                        background[i] if isinstance(background, list) else background,
                        result.resize((width, height)),
                        cropped_inpaint_position
                    )
                    for i, result in enumerate(results)
                ]
            elif results:
                images = results

            # Execute SVD, if requested
            images, nsfw = self.execute_stable_video(
                pipeline,
                images=images,
                nsfw=nsfw,
                task_callback=task_callback,
                progress_callback=progress_callback,
                image_callback=image_callback,
                image_callback_steps=image_callback_steps,
                invocation_kwargs=invocation_kwargs
            )

            # Execute detailer, if requested
            images, nsfw = self.execute_detailer(
                pipeline,
                images=images,
                nsfw=nsfw,
                no_inference=no_inference,
                task_callback=task_callback,
                progress_callback=progress_callback,
                image_callback=image_callback,
                image_callback_steps=image_callback_steps,
                invocation_kwargs=invocation_kwargs
            )

            # Execute upscale, if requested
            images, nsfw = self.execute_upscale(
                pipeline,
                images=images,
                nsfw=nsfw,
                no_inference=no_inference,
                task_callback=task_callback,
                progress_callback=progress_callback,
                image_callback=image_callback,
                image_callback_steps=image_callback_steps,
                invocation_kwargs=invocation_kwargs
            )

            # Execute interpolation/reflect, if requested
            result = self.execute_interpolate(
                pipeline,
                images=images,
                nsfw=nsfw,
                task_callback=task_callback,
                progress_callback=progress_callback,
                image_callback=image_callback,
                image_callback_steps=image_callback_steps,
            )
            return result
        finally:
            pipeline.safe = safe_status # Restore safety if it was set
            pipeline.clear_task_callback() # Unregister this callback
            pipeline.stop_keepalive() # Make sure this is stopped
            pipeline.clear_memory() # Clear memory

    def prepare_pipeline(self, pipeline: DiffusionPipelineManager, ignore_animation: bool=False) -> None:
        """
        Assigns pipeline-level variables.
        """
        pipeline.start_keepalive() # Make sure this is going

        if self.animation_frames is not None and self.animation_frames > 0 and self.animation_engine != "svd" and not ignore_animation:
            pipeline.animator = self.model
            pipeline.animator_vae = self.vae # type: ignore[assignment]
            pipeline.frame_window_size = self.frame_window_size # type: ignore[assignment]
            pipeline.frame_window_stride = self.frame_window_stride
            pipeline.position_encoding_truncate_length = self.position_encoding_truncate_length
            pipeline.position_encoding_scale_length = self.position_encoding_scale_length
            pipeline.motion_module = self.motion_module
        else:
            pipeline.model = self.model # type: ignore[assignment]
            pipeline.vae = self.vae # type: ignore[assignment]

        pipeline.tiling_size = self.tiling_size
        pipeline.tiling_stride = self.tiling_stride # type: ignore[assignment]
        pipeline.tiling_mask_type = self.tiling_mask_type # type: ignore[attr-defined,assignment]

        pipeline.refiner = self.refiner
        pipeline.refiner_vae = self.refiner_vae # type: ignore[assignment]

        pipeline.inpainter = self.inpainter
        pipeline.inpainter_vae = self.inpainter_vae # type: ignore[assignment]

        pipeline.lora = self.lora # type: ignore[assignment]
        pipeline.lycoris = self.lycoris # type: ignore[assignment]
        pipeline.inversion = self.inversion # type: ignore[assignment]

        if (
            self.scheduler_beta_start is not None or
            self.scheduler_beta_end is not None or
            self.scheduler_beta_schedule is not None
        ):
            scheduler_config: Dict[str, Any] = {}
            if self.scheduler_beta_start is not None:
                scheduler_config["beta_start"] = self.scheduler_beta_start
            if self.scheduler_beta_end is not None:
                scheduler_config["beta_end"] = self.scheduler_beta_end
            if self.scheduler_beta_schedule:
                scheduler_config["beta_schedule"] = self.scheduler_beta_schedule
            pipeline.scheduler_config = scheduler_config

        pipeline.scheduler = self.scheduler # type: ignore[assignment]

        if self.build_tensorrt:
            pipeline.build_tensorrt = True

        if self.safe is not None:
            pipeline.safe = self.safe # safety checking override

    def execute_inference(
        self,
        pipeline: DiffusionPipelineManager,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Image], List[bool]]:
        """
        Executes primary inference
        """
        from PIL import Image, ImageDraw

        # Define progress and latent callback kwargs
        callback_kwargs = {
            "progress_callback": progress_callback,
            "latent_callback_steps": image_callback_steps,
            "latent_callback_type": "pil",
            "task_callback": task_callback
        }

        if self.seed is not None:
            # Set up the RNG
            pipeline.seed = self.seed

        total_images = self.iterations
        if self.animation_frames is not None and self.animation_frames > 0:
            total_images *= self.animation_frames
        else:
            total_images *= self.samples

        width = pipeline.size if self.width is None else self.width
        height = pipeline.size if self.height is None else self.height

        if "image" in invocation_kwargs:
            if isinstance(invocation_kwargs["image"], list):
                images = [
                    invocation_kwargs["image"][i].copy() if len(invocation_kwargs["image"]) > i else invocation_kwargs["image"][-1].copy()
                    for i in range(total_images)
                ]
                # Replace empty space with noise (instead of black)
                for i, image in enumerate(invocation_kwargs["image"]):
                    invocation_kwargs["image"][i] = self.remove_alpha(image, pipeline)
            else:
                images = [
                    invocation_kwargs["image"].copy()
                    for i in range(total_images)
                ]
                invocation_kwargs["image"] = self.remove_alpha(invocation_kwargs["image"], pipeline)
        else:
            images = [
                Image.new("RGBA", (width, height))
                for i in range(total_images)
            ]

        image_draw = [
            ImageDraw.Draw(image)
            for image in images
        ]
        nsfw_content_detected = [False] * total_images

        # Trigger the callback with base images after scaling and processing
        if image_callback is not None and invocation_kwargs.get("image", None):
            image_callback(images)

        # Determine what controlnets to use
        controlnets = (
            None if not invocation_kwargs.get("control_images", None)
            else list(invocation_kwargs["control_images"].keys())
        )

        repeat_images = self.animation_frames is not None and self.animation_engine == "svd"
        expected_iteration_length = self.samples
        if self.animation_frames is not None:
            expected_iteration_length *= self.animation_frames

        for it in range(self.iterations):
            image_index_start = (it * self.samples)
            if image_callback is not None:
                def iteration_image_callback(callback_images: List[Image]) -> None:
                    """
                    Wrap the original image callback so we're actually pasting the initial image on the main canvas
                    """
                    for j, callback_image in enumerate(callback_images):
                        image_index = image_index_start + j
                        images[image_index] = callback_image
                    if repeat_images:
                        callback_count = len(callback_images)
                        repeat_count = expected_iteration_length - callback_count
                        for k in range(repeat_count):
                            images[image_index_start + callback_count + k] = images[image_index_start + callback_count - 1]
                    image_callback(images)  # type: ignore
            else:
                iteration_image_callback = None  # type: ignore

            if invocation_kwargs.get("animation_frames", None):
                pipeline.animator_controlnets = controlnets # type: ignore[assignment]
            elif invocation_kwargs.get("mask", None):
                pipeline.inpainter_controlnets = controlnets # type: ignore[assignment]
            else:
                pipeline.controlnets = controlnets # type: ignore[assignment]

            result = pipeline(
                latent_callback=iteration_image_callback,
                **invocation_kwargs,
                **callback_kwargs # type: ignore[arg-type]
            )

            for j, image in enumerate(result["images"]):
                images[image_index_start + j] = image
                nsfw_content_detected[image_index_start + j] = nsfw_content_detected[image_index_start + j] or (
                    "nsfw_content_detected" in result and result["nsfw_content_detected"][j]
                )

            if repeat_images:
                returned_count = len(result["images"])
                repeat_count = expected_iteration_length - returned_count
                for k in range(repeat_count):
                    images[image_index_start + returned_count + k] = images[image_index_start + returned_count - 1]

            # Call the callback
            if image_callback is not None:
                image_callback(images)

        return images, nsfw_content_detected

    def execute_stable_video(
        self,
        pipeline: DiffusionPipelineManager,
        images: List[Image],
        nsfw: List[bool],
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Image], List[bool]]:
        """
        Executes stable video diffusion if required
        """
        if nsfw[0] or not self.animation_frames or self.animation_engine != "svd":
            return images, nsfw

        # Set up RNG
        if self.seed is not None:
            pipeline.seed = self.seed

        svd_kwargs = {
            "fps": self.fps,
            "image": images[0],
            "num_frames": self.animation_frames,
            "decode_chunk_size": self.frame_decode_chunk_size,
            "num_inference_steps": 25, # self.num_inference_steps,
            "min_guidance_scale": self.min_guidance_scale,
            "max_guidance_scale": self.max_guidance_scale,
            "noise_aug_strength": self.noise_aug_strength,
            "motion_bucket_id": self.motion_bucket_id
        }

        if self.motion_vectors or self.optical_flow:
            svd_kwargs["motion_vectors"] = self.motion_vectors
            svd_kwargs["motion_vector_repeat_window"] = self.motion_vector_repeat_window
            svd_kwargs["gaussian_sigma"] = self.gaussian_sigma
            if self.optical_flow:
                optical_flow = {}
                for method in self.optical_flow:
                    frame_lists = self.optical_flow[method]
                    if not frame_lists:
                        continue
                    if not isinstance(frame_lists[0], list):
                        frame_lists = [frame_lists]
                    optical_flow[method] = [
                        [
                            self.prepare_image(
                                width=self.width,
                                height=self.height,
                                image=frame,
                                animation_frames=self.animation_frames,
                                return_mask=False
                            )
                            for frame in frame_list
                        ]
                        for frame_list in frame_lists
                    ]
                svd_kwargs["optical_flow"] = optical_flow
            logger.debug(f"Calling DragNUWA pipeline with arguments {redact_for_log(svd_kwargs)}")
            frames = pipeline.dragnuwa_img2vid(
                task_callback=task_callback,
                progress_callback=progress_callback,
                image_callback=image_callback,
                image_callback_steps=image_callback_steps,
                **svd_kwargs
            )
        else:
            svd_kwargs["use_xt"] = self.svd_model == "svd_xt"
            logger.debug(f"Calling SVD pipeline with arguments {redact_for_log(svd_kwargs)}")
            frames = pipeline.svd_img2vid(
                task_callback=task_callback,
                progress_callback=progress_callback,
                image_callback=image_callback,
                image_callback_steps=image_callback_steps,
                **svd_kwargs
            )

        return frames, [False] * len(frames)
        
    def execute_detailer(
        self,
        pipeline: DiffusionPipelineManager,
        images: List[Image],
        nsfw: List[bool],
        no_inference: bool = False,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Image], List[bool]]:
        """
        Runs the after detailer.
        """
        # Guard clause, make sure detailing was requested.
        if not self.detailer_face_restore and not self.detailer_face_inpaint and not self.detailer_hand_inpaint:
            return images, nsfw

        from PIL import Image, ImageFilter
        from enfugue.diffusion.util import Video
        from enfugue.diffusion.util.prompt_util import Prompt

        isolated_detail_masks = []
        isolated_bounding_boxes = []
        detail_masks = []

        width, height = images[0].size

        black_mask = Image.new("L", (width, height))
        white_mask = Image.new("L", (width, height), (255))

        # First get detailing masks; we'll use them for pasting the face fix and inpainting
        with pipeline.control_image_processor.pose_detector.best() as pose_detector:
            for i, image in enumerate(images):
                isolated_detail_masks.append(
                    [
                        this_mask.convert("L")
                        for this_mask in pose_detector.detail_mask( # type: ignore[attr-defined]
                            image,
                            include_hands=self.detailer_hand_inpaint,
                            include_face=self.detailer_face_inpaint or self.detailer_face_restore,
                            isolated=True
                        )
                    ]
                )
                isolated_bounding_boxes.append([])
                detail_masks.append(black_mask.copy())
                for isolated_mask in isolated_detail_masks[-1]:
                    detail_masks[-1].paste(white_mask, mask=isolated_mask)
                    isolated_bounding_boxes[-1].append(self.get_inpaint_bounding_box(
                        isolated_mask,
                        size=self.detailer_inpaint_min_size,
                        feather=self.detailer_inpaint_feather
                    ))

        # Check and perform face restore pass
        if self.detailer_face_restore:
            with pipeline.upscaler.face_restore() as restore:
                if task_callback is not None:
                    task_callback("Restoring Faces")
                if progress_callback is not None:
                    progress_callback(0, len(images), 0.0)

                for i, image in enumerate(images):
                    restore_start = datetime.now()
                    paste_mask = dilate_erode(detail_masks[i], 16)
                    paste_mask = paste_mask.filter(ImageFilter.BoxBlur(self.detailer_inpaint_blur)) # type: ignore[union-attr]
                    images[i].paste(restore(image), mask=paste_mask.convert("L"))
                    if progress_callback is not None:
                        restore_time = (datetime.now() - restore_start).total_seconds()
                        progress_callback(i, len(images), 1/restore_time)

            if image_callback is not None:
                image_callback(images)

        # Guard clause, if no inpaint requested go ahead and return
        if not (self.detailer_face_inpaint or self.detailer_hand_inpaint) and self.detailer_inpaint_strength:
            return images, nsfw

        # We will be inpainting, first assemble our arguments
        if self.prompts:
            prompts = [
                Prompt(
                    positive=self.merge_prompts(
                        (DEFAULT_UPSCALE_PROMPT, 1.0),
                        (prompt["positive"], GLOBAL_PROMPT_UPSCALE_WEIGHT),
                        (self.model_prompt, MODEL_PROMPT_WEIGHT),
                        (self.refiner_prompt, MODEL_PROMPT_WEIGHT),
                    ),
                    positive_2=None if not prompt.get("positive_2", None) else self.merge_prompts(
                        (prompt["positive_2"], GLOBAL_PROMPT_UPSCALE_WEIGHT),
                        (self.model_prompt_2, MODEL_PROMPT_WEIGHT),
                        (self.refiner_prompt_2, MODEL_PROMPT_WEIGHT),
                    ),
                    negative=self.merge_prompts(
                        (prompt.get("negative", None), GLOBAL_PROMPT_UPSCALE_WEIGHT),
                        (self.model_negative_prompt, MODEL_PROMPT_WEIGHT),
                        (self.refiner_negative_prompt, MODEL_PROMPT_WEIGHT),
                    ),
                    negative_2=None if not prompt.get("negative_2", None) else self.merge_prompts(
                        (prompt["negative_2"], GLOBAL_PROMPT_UPSCALE_WEIGHT),
                        (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                        (self.refiner_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                    )
                )
                for prompt in self.prompts
            ]
            prompt, prompt_2, negative_prompt, negative_prompt_2 = None, None, None, None
        else:
            prompts = None
            prompt = self.merge_prompts( # type: ignore[assignment]
                (DEFAULT_UPSCALE_PROMPT, 1.0),
                (self.prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_prompt, MODEL_PROMPT_WEIGHT),
                (self.refiner_prompt, MODEL_PROMPT_WEIGHT),
            )
            prompt_2 = self.merge_prompts(
                (self.prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_prompt_2, MODEL_PROMPT_WEIGHT),
                (self.refiner_prompt_2, MODEL_PROMPT_WEIGHT),
            )
            negative_prompt = self.merge_prompts(
                (self.negative_prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_negative_prompt, MODEL_PROMPT_WEIGHT),
                (self.refiner_negative_prompt, MODEL_PROMPT_WEIGHT),
            )
            negative_prompt_2 = self.merge_prompts(
                (self.negative_prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                (self.refiner_negative_prompt_2, MODEL_PROMPT_WEIGHT),
            )

        guidance_scale = self.guidance_scale if self.detailer_guidance_scale is None else self.detailer_guidance_scale
        num_inference_steps = self.num_inference_steps if self.detailer_inference_steps is None else self.detailer_inference_steps

        # Check if we didn't perform inference before now; if so, we need to set up the pipeline
        if no_inference:
            self.prepare_pipeline(pipeline)

        # Determine which pipeline we're going to be using
        use_inpainter = self.detailer_switch_pipeline or invocation_kwargs.get("mask", None) and not self.animation_frames
        #inpainter_ip_adapter = "plus-face-id"
        inpainter_ip_adapter = None
        use_inpainter = False

        if use_inpainter:
            inpaint_size = 1024 if pipeline.inpainter_is_sdxl else 512
            if self.detailer_controlnet:
                pipeline.inpainter_controlnets = self.detailer_controlnet # type: ignore[assignment]
            else:
                pipeline.inpainter_controlnets = None # type: ignore[assignment]
            if inpainter_ip_adapter is not None and getattr(pipeline, "_ip_adapter_model", None) != inpainter_ip_adapter:
                pipeline.unload_inpainter("Setting IP Adapter Model")
                pipeline._ip_adapter_model = inpainter_ip_adapter
            detail_pipeline = pipeline.inpainter_pipeline # type: ignore[assignment]
        elif self.animation_frames:
            inpaint_size = 1024 if pipeline.animator_is_sdxl else 512
            if self.detailer_controlnet:
                pipeline.animator_controlnets = self.detailer_controlnet # type: ignore[assignment]
            else:
                pipeline.animator_controlnets = None # type: ignore[assignment]
            if inpainter_ip_adapter is not None and getattr(pipeline, "_ip_adapter_model", None) != inpainter_ip_adapter:
                pipeline.unload_animator("Setting IP Adapter Model")
                pipeline._ip_adapter_model = inpainter_ip_adapter
            detail_pipeline = pipeline.animator_pipeline # type: ignore[assignment]
        else:
            inpaint_size = 1024 if pipeline.is_sdxl else 512
            if self.detailer_controlnet:
                pipeline.controlnets = self.detailer_controlnet # type: ignore[assignment]
            else:
                pipeline.controlnets = None # type: ignore[assignment]
            if inpainter_ip_adapter is not None and getattr(pipeline, "_ip_adapter_model", None) != inpainter_ip_adapter:
                pipeline.unload_pipeline("Setting IP Adapter Model")
                pipeline._ip_adapter_model = inpainter_ip_adapter
            detail_pipeline = pipeline.pipeline # type: ignore[assignment]

        # If requested, store control images for each sample as well
        control_images = []
        if self.detailer_controlnet and self.detailer_controlnet_scale:
            with pipeline.control_image_processor.processor(self.detailer_controlnet) as process:
                processed_control_images = []
                for i, image in enumerate(images):
                    processed_control_images.append(process(image))
                control_images = [
                    dict([ # type: ignore[misc]
                        (self.detailer_controlnet, [(processed_image, self.detailer_controlnet_scale)])
                    ])
                    for processed_image in processed_control_images
                ]

        # Assemble groupings for detail passes in animation, or create dummy groups
        frame_box_mask_groups = [] # (start_frame, bounding_box, masks)
        if self.animation_frames:
            grouped_boxes = self.group_bounding_boxes(
                self.width,
                self.height,
                isolated_bounding_boxes
            )
            logger.info(f"GROUPS {grouped_boxes}")
            logger.info("MASKS")
            for f, frame in enumerate(isolated_detail_masks):
                fl = len(frame)
                logger.info(f"{f}/{fl}")
            for g, group in enumerate(grouped_boxes):
                consecutive_hidden_frames = 0
                current_frame_start = 0
                current_frames = []

                def maybe_add_frames():
                    nonlocal current_frames, current_frame_start
                    current_frames = current_frames[:max([i for i, v in enumerate(current_frames) if v]+[0])+1]
                    if current_frames and len(current_frames) >= 4:
                        current_masks = []
                        x0, y0, x1, y1 = -1, -1, -1, -1
                        logger.info(f"start {current_frame_start}")
                        for i, this_frame in enumerate(current_frames):
                            if current_frame_start + i >= len(isolated_detail_masks):
                                break
                            if this_frame is None:
                                current_masks.append(Image.new("L", (width, height)))
                                continue
                            mask_index, bounding_box = this_frame
                            logger.info("index {0} maskindex {1}".format(current_frame_start+i, mask_index))
                            if mask_index >= len(isolated_detail_masks[current_frame_start+i]):
                                current_masks.append(Image.new("L", (width, height)))
                            else:
                                current_masks.append(
                                    isolated_detail_masks[current_frame_start+i][mask_index]
                                )
                            (bx0, by0), (bx1, by1) = bounding_box
                            x0 = bx0 if x0 == -1 else min(x0, bx0)
                            x1 = bx1 if x1 == -1 else max(x1, bx1)
                            y0 = by0 if y0 == -1 else min(y0, by0)
                            y1 = by1 if y1 == -1 else max(y1, by1)

                        frame_box_mask_groups.append((
                            current_frame_start,
                            ((x0, y0), (x1, y1)),
                            current_masks
                        ))
                        current_frames = []

                for index, tup in enumerate(group):
                    if tup is None and current_frames:
                        consecutive_hidden_frames += 1
                    else:
                        consecutive_hidden_frames = 0
                    if current_frames or tup:
                        current_frames.append(tup)
                    else:
                        current_frame_start = index + 1
                    if consecutive_hidden_frames >= 4:
                        # Break
                        maybe_add_frames()
                        current_frame_start = index + 1

                # Add the last group of frames if found
                maybe_add_frames()
        else:
            # Create dummy groups so we can share code between animation and images
            for i in range(len(isolated_detail_masks)):
                for box, mask in zip(isolated_bounding_boxes[i], isolated_detail_masks[i]):
                    # See if we can merge this box/mask
                    merged = False
                    for j, (index, other_box, other_mask) in enumerate(frame_box_mask_groups):
                        overlap = self.find_overlap(box, other_box)
                        if overlap:
                            (x00, y00), (x01, y01) = box
                            (x10, y10), (x11, y11) = other_box
                            (ox0, oy0), (ox1, oy1) = overlap
                            overlap_area = (ox1 - ox0) * (oy1 - oy0)
                            box_area = (x01 - x00) * (y01 - y00)
                            other_box_area = (x11 - x10) * (y11 - y10)
                            if overlap_area > min(box_area, other_box_area) * 0.2:
                                merged = True
                                new_bbox = [(min(x00, x10), min(y00, y10)), (max(x01, x11), max(y01, y11))]
                                white = Image.new(other_mask.mode, other_mask.size, (255,) if other_mask.mode == "L" else (255,255,255))
                                other_mask.paste(white, mask=mask)
                                frame_box_mask_groups[j] = (index, new_bbox, other_mask)
                    if not merged:
                        frame_box_mask_groups.append((i, box, mask))

        # Perform detail passes
        total_detail_passes = len(frame_box_mask_groups)
        for i, (frame_start_index, bounding_box, mask) in enumerate(frame_box_mask_groups):
            if nsfw[frame_start_index]:
                continue

            if isinstance(mask, list):
                mask_max, mask_min = -1, -1
                for mask_img in mask:
                    mask_img_max, mask_img_min = mask_img.getextrema()
                    mask_max = mask_img_max if mask_max == -1 else max(mask_img_max, mask_max)
                    mask_min = mask_img_min if mask_min == -1 else min(mask_img_min, mask_min)
            else:
                mask_max, mask_min = mask.getextrema()

            if mask_max == mask_min == 0:
                logger.debug(f"No detailable areas found in pass {i+1}, skipping.")
                continue

            if task_callback is not None:
                task_callback(f"Detailing pass {i+1}/{total_detail_passes}")

            (x0, y0), (x1, y1) = bounding_box
            extend_up = (y1 - y0) // 3
            dilate_amount = max(self.detailer_inpaint_feather, int(self.detailer_inpaint_dilate_ratio * max((x1 - x0), (y1 - y0))))
            y0 = max(y0 - extend_up, 0)

            """
            x0 = max(x0 - dilate_amount // 2, 0)
            y0 = max(y0 - dilate_amount // 2, 0)
            x1 = max(x1 + dilate_amount // 2, width - 1)
            y1 = max(y1 + dilate_amount // 2, height - 1)
            """

            if isinstance(mask, list):
                isolated_mask = [
                    dilate_erode(
                        self.extend_mask(mask_img, extend_up),
                        dilate_amount
                    )
                    for mask_img in mask
                ]
            else:
                isolated_mask = dilate_erode(
                    self.extend_mask(mask, extend_up),
                    dilate_amount
                )

            num_frame_masks = len(isolated_mask) if isinstance(isolated_mask, list) else 1
            frame_end_index = frame_start_index + num_frame_masks
            added_before = 0
            added_after = 0

            if num_frame_masks > 1:
                logger.info(f"START: {frame_start_index}:{frame_end_index}")
                frame_window_size = self.frame_window_size if self.frame_window_size else self.animation_frames
                if frame_window_size and num_frame_masks < frame_window_size:
                    frames_to_add = frame_window_size - num_frame_masks # 64 - 25 = 39
                    added_after = frames_to_add
                    frame_end_index = frame_start_index + num_frame_masks + frames_to_add
                    if frame_end_index >= self.animation_frames:
                        to_shift = min(frame_end_index - self.animation_frames, frame_start_index)
                        frame_start_index -= to_shift
                        frame_end_index -= to_shift
                        added_before += to_shift
                        added_after -= to_shift
                if self.frame_window_size and self.frame_window_stride:
                    # Make sure it's divisible by stride
                    actual_frames = frame_end_index - frame_start_index
                    stride_frames = (actual_frames // self.frame_window_stride) * self.frame_window_stride
                    leftover = stride_frames - actual_frames
                    logger.debug(f"ADDING {leftover} {frame_end_index} {frame_start_index} {frame_end_index-frame_start_index} {self.frame_window_stride}?")
                    if leftover != 0:
                        frame_end_index += leftover
                        added_after += leftover
                        if frame_end_index >= self.animation_frames:
                            remainder = frame_end_index - self.animation_frames
                            frame_end_index = self.animation_frames
                            added_after -= remainder
                            if remainder < frame_start_index:
                                frame_start_index -= remainder
                                added_before += remainder

            logger.info(f"END: {frame_start_index}:{frame_end_index} ({added_before} before, {added_after} after)")
            isolated_control_images = None
            if len(control_images) > frame_start_index:
                isolated_control_images = control_images
                if not isinstance(mask, list):
                    isolated_control_images = isolated_control_images[frame_start_index]
                else:
                    controlnet_name = next(isolated_control_images[frame_start_index].keys())
                    scale = isolated_control_images[frame_start_index][controlnet_name][1]
                    controlnet_images = []

                    for controlnet_dict in isolated_control_images[frame_start_index:frame_end_index]:
                        controlnet_images.append(controlnet_dict[controlnet_name][0])

                    isolated_control_images = dict([
                        (controlnet_name, (controlnet_images, scale))
                    ])

            if isinstance(isolated_mask, list):
                isolated_image = [
                    img.crop((x0, y0, x1, y1))
                    for img in images[frame_start_index:frame_end_index]
                ]
            else:
                isolated_image = images[frame_start_index].crop((x0, y0, x1, y1))

            if isinstance(isolated_mask, list):
                isolated_mask = [
                    mask_img.crop((x0, y0, x1, y1))
                    for mask_img in isolated_mask
                ]
                isolated_mask = (
                    ([Image.new("L", isolated_mask[0].size)] * added_before) +
                    isolated_mask +
                    ([Image.new("L", isolated_mask[0].size)] * added_after)
                )
            else:
                isolated_mask = isolated_mask.crop((x0, y0, x1, y1))

            if isolated_control_images:
                for controlnet_name in isolated_control_images:
                    isolated_control_images[controlnet_name] = [
                        (image.crop((x0, y0, x1, y1)), scale)
                        for (image, scale) in isolated_control_images[controlnet_name]
                    ]

            original_width = x1 - x0
            original_height = y1 - y0
            inference_width = original_width
            inference_height = original_height

            scale_factor = (inpaint_size / max(original_width, original_height)) + self.detailer_upscale_amount

            if scale_factor > 1:
                inference_width *= scale_factor
                inference_height *= scale_factor
                inference_width = int((inference_width // 8) * 8)
                inference_height = int((inference_height // 8) * 8)

                if isinstance(isolated_image, list):
                    isolated_image = [
                        img.resize((inference_width, inference_height), Image.LANCZOS)
                        for img in isolated_image
                    ]
                else:
                    isolated_image = isolated_image.resize((inference_width, inference_height), Image.LANCZOS)

                if isinstance(isolated_mask, list):
                    isolated_mask = [
                        mask_img.resize((inference_width, inference_height), Image.NEAREST)
                        for mask_img in isolated_mask
                    ]
                else:
                    isolated_mask = isolated_mask.resize((inference_width, inference_height), Image.NEAREST)

                if isolated_control_images is not None:
                    for controlnet_name in isolated_control_images:
                        isolated_control_images[controlnet_name] = [
                            (image.resize((inference_width, inference_height), Image.LANCZOS), scale)
                            for (image, scale) in isolated_control_images[controlnet_name]
                        ]

            detail_kwargs = {
                "width": inference_width,
                "height": inference_height,
                "image": isolated_image,
                "mask": isolated_mask,
                "prompts": prompts,
                "prompt": prompt,
                "prompt_2": prompt_2,
                "negative_prompt": negative_prompt,
                "negative_prompt_2": negative_prompt_2,
                "prompts": self.prompts if self.prompts and len(self.prompts) > 1 else None,
                "control_images": isolated_control_images,
                "generator": pipeline.generator,
                "noise_generator": pipeline.noise_generator,
                "device": pipeline.device,
                "offload_models": pipeline.pipeline_sequential_onload,
                "strength": self.detailer_inpaint_strength,
                "num_results_per_prompt": 1,
                "progress_callback": progress_callback,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "animation_frames": None if not isinstance(isolated_image, list) else len(isolated_image),
                "frame_window_size": self.frame_window_size,
                "frame_window_stride": self.frame_window_stride,
                "motion_scale": invocation_kwargs.get("motion_scale", None),
                "loop": invocation_kwargs.get("loop", False),
                "progress_callback": progress_callback,
                "latent_callback_type": "pil",
            }

            detail_kwargs["ip_adapter_images"] = invocation_kwargs.get("ip_adapter_images", None)
            """
            # If the pipeline has an IP adapter, pass the image through that
            if detail_pipeline.ip_adapter_loaded:
                # Extend the image crop up and to the sides to get hair for face ID
                ix0 = max(x0 - 16, 0)
                ix1 = min(x1 + 16, width-1)
                iy0 = max(y0 - 32, 0)
                iy1 = min(y1 + 8, height-1)
                if isinstance(isolated_mask, list):
                    ip_adapter_image = [
                        image.crop((ix0, iy0, ix1, iy1))
                        for image in images
                    ]
                else:
                    ip_adapter_image = images[frame_start_index].crop((ix0, iy0, ix1, iy1))
                logger.debug(f"Detail pipeline has IP adapter loaded, adding image to adapter input.")
                detail_kwargs["ip_adapter_images"] = [(ip_adapter_image, self.detailer_inpaint_ip_adapter_scale)]
            """
            inpaint_image_callback = None
            if image_callback is not None and image_callback_steps:
                # Create resized callback
                def resized_callback(inference_images: List[Image]) -> None:
                    """
                    Paste the images then callback.
                    """
                    callback_images = [image.copy() for image in images]
                    for j, result in enumerate(inference_images):
                        callback_images[j] = self.paste_inpaint_image(
                            callback_images[j],
                            result.resize((original_width, original_height), Image.NEAREST),
                            (x0, y0, x1, y1),
                            inpaint_feather=self.detailer_inpaint_feather
                        )
                    image_callback(callback_images) # type: ignore

                inpaint_image_callback = resized_callback

            logger.debug(f"Detailing pass {i} starting from image {frame_start_index+1} through {frame_end_index} with arguments {detail_kwargs}")

            pipeline.stop_keepalive()
            results = detail_pipeline( # type: ignore
                latent_callback=inpaint_image_callback,
                latent_callback_steps=image_callback_steps,
                **detail_kwargs
            )["images"] # type: ignore

            for j, result in enumerate(results):
                images[frame_start_index+j] = self.paste_inpaint_image(
                    images[frame_start_index+j],
                    result.resize((original_width, original_height), Image.NEAREST),
                    (x0, y0, x1, y1),
                    inpaint_feather=self.detailer_inpaint_feather
                )

            if image_callback is not None:
                image_callback(images)

        return images, nsfw

    def execute_upscale(
        self,
        pipeline: DiffusionPipelineManager,
        images: List[Image],
        nsfw: List[bool],
        no_inference: bool = False,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Image], List[bool]]:
        """
        Executes upscale steps
        """
        from diffusers.utils.pil_utils import PIL_INTERPOLATION
        animation_frames = invocation_kwargs.get("animation_frames", None)

        for upscale_step in self.upscale_steps:
            method = upscale_step["method"]
            amount = upscale_step["amount"]
            num_inference_steps = upscale_step.get("num_inference_steps", DEFAULT_UPSCALE_INFERENCE_STEPS)
            guidance_scale = upscale_step.get("guidance_scale", DEFAULT_UPSCALE_GUIDANCE_SCALE)
            prompt = upscale_step.get("prompt", DEFAULT_UPSCALE_PROMPT)
            prompt_2 = upscale_step.get("prompt_2", None)
            negative_prompt = upscale_step.get("negative_prompt", None)
            negative_prompt_2 = upscale_step.get("negative_prompt_2", None)
            strength = upscale_step.get("strength", None)
            controlnet = upscale_step.get("controlnet", None)
            controlnet_scale = upscale_step.get("controlnet_scale", 1.0)
            scheduler = upscale_step.get("scheduler", self.scheduler)
            tiling_unet = upscale_step.get("tiling_unet", True)
            tiling_vae = upscale_step.get("tiling_vae", True)
            tiling_stride = upscale_step.get("tiling_stride", DEFAULT_UPSCALE_TILING_STRIDE)
            tiling_size = upscale_step.get("tiling_size", DEFAULT_UPSCALE_TILING_SIZE)
            tiling_mask_type = upscale_step.get("tiling_mask_type", None)
            tiling_mask_kwargs = upscale_step.get("tiling_mask_kwargs", None)
            frame_window_size = upscale_step.get("frame_window_size", self.frame_window_size)
            frame_window_stride = upscale_step.get("frame_window_stride", self.frame_window_stride)
            noise_offset = upscale_step.get("noise_offset", None)
            noise_method = upscale_step.get("noise_method", None)
            noise_blend_method = upscale_step.get("noise_blend_method", None)
            refiner = self.refiner is not None and upscale_step.get("refiner", True)
            
            prompt = self.merge_prompts( # type: ignore[assignment]
                (prompt, 1.0),
                (self.prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_prompt, MODEL_PROMPT_WEIGHT),
                (self.refiner_prompt, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict["positive"], GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            prompt_2 = self.merge_prompts(
                (prompt_2, 1.0),
                (self.prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_prompt_2, MODEL_PROMPT_WEIGHT),
                (self.refiner_prompt_2, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict.get("positive_2", None), GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            negative_prompt = self.merge_prompts(
                (negative_prompt, 1.0),
                (self.negative_prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_negative_prompt, MODEL_PROMPT_WEIGHT),
                (self.refiner_negative_prompt, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict.get("negative", None), GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            negative_prompt_2 = self.merge_prompts(
                (negative_prompt_2, 1.0),
                (self.negative_prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                (self.refiner_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict.get("negative_2", None), GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            @contextmanager
            def get_upscale_image() -> Iterator[Callable[[Image], Image]]:
                if method in ["esrgan", "esrganime", "gfpgan", "ccsr"]:
                    if method == "ccsr":
                        pipeline.unload_all("clearing memory for upscaler")
                    elif refiner:
                        pipeline.unload_pipeline("clearing memory for upscaler")
                        pipeline.unload_inpainter("clearing memory for upscaler")
                        pipeline.offload_refiner()
                    else:
                        pipeline.offload_pipeline()
                        pipeline.offload_animator()
                        pipeline.offload_inpainter()
                        pipeline.unload_refiner("clearing memory for upscaler")
                    if method == "ccsr":
                        with pipeline.upscaler.ccsr() as upscale:
                            def execute_upscale(image: Image) -> Image:
                                return upscale( # type: ignore[call-arg]
                                    image,
                                    outscale=amount,
                                    tile_diffusion_stride=256,
                                    tile_diffusion_size=None if not tiling_unet else 512,
                                    tile_vae_encode_size=None if not tiling_vae else 512,
                                    tile_vae_decode_size=None if not tiling_vae else 128,
                                    progress_callback=progress_callback,
                                )
                            yield execute_upscale
                    elif method == "gfpgan":
                        with pipeline.upscaler.gfpgan(tile=512) as upscale:
                            def execute_upscale(image: Image) -> Image:
                                return upscale(image, outscale=amount) # type: ignore[call-arg]
                            yield execute_upscale
                    else:
                        with pipeline.upscaler.esrgan(tile=512, anime=method=="esrganime") as upscale:
                            def execute_upscale(image: Image) -> Image:
                                return upscale(image, outscale=amount) # type: ignore[call-arg]
                            yield execute_upscale
                elif method in PIL_INTERPOLATION:
                    def pil_resize(image: Image) -> Image:
                        image_width, image_height = image.size
                        return image.resize(
                            (int(image_width * amount), int(image_height * amount)),
                            resample=PIL_INTERPOLATION[method]
                        )
                    yield pil_resize
                else:
                    logger.error(f"Unknown method {method}")
                    def no_resize(image: Image) -> Image:
                        return image
                    yield no_resize

            if task_callback:
                task_callback(f"Preparing upscale model")

            with get_upscale_image() as upscale_image:
                pipeline.stop_keepalive()
                if task_callback:
                    task_callback(f"Upscaling samples")
                if progress_callback is not None:
                    progress_callback(0, len(images), 0.0)
                for i, image in enumerate(images):
                    upscale_start = datetime.now()
                    if nsfw is not None and nsfw[i]:
                        continue
                    logger.debug(f"Upscaling sample {i} by {amount} using {method}")
                    if method == "ccsr" and task_callback:
                        task_callback(f"Upscaling sample {i+1}")
                    images[i] = upscale_image(image)
                    upscale_time = max((datetime.now() - upscale_start).total_seconds(), 1e-5)
                    if progress_callback:
                        progress_callback(i+1, len(images), 1/upscale_time)

            if image_callback:
                image_callback(images)

            if strength is not None and strength > 0:
                if task_callback:
                    task_callback("Preparing upscale pipeline")

                if no_inference:
                    self.prepare_pipeline(pipeline)

                if refiner:
                    # Refiners have safety disabled from the jump
                    logger.debug("Using refiner for upscaling.")
                    re_enable_safety = False
                    tiling_size = max(tiling_size, pipeline.refiner_size) # type: ignore
                    tiling_stride = min(tiling_stride, pipeline.refiner_size // 2) # type: ignore
                else:
                    # Disable pipeline safety here, it gives many false positives when upscaling.
                    # We'll re-enable it after.
                    logger.debug("Using base pipeline for upscaling.")
                    re_enable_safety = pipeline.safe
                    if animation_frames:
                        tiling_size = max(tiling_size, pipeline.animator_size) # type: ignore
                        tiling_stride = min(tiling_stride, pipeline.animator_size // 2) # type: ignore
                    else:
                        tiling_size = max(tiling_size, pipeline.size) # type: ignore
                        tiling_stride = min(tiling_stride, pipeline.size // 2) # type: ignore
                    pipeline.safe = False

                if scheduler is not None:
                    pipeline.scheduler = scheduler # type: ignore[assignment]

                if animation_frames:
                    upscaled_images = [images]
                else:
                    upscaled_images = images

                for i, image in enumerate(upscaled_images):
                    if nsfw is not None and nsfw[i]:
                        continue

                    if isinstance(image, list):
                        width, height = image[0].size
                    else:
                        width, height = image.size

                    kwargs = {
                        "width": width,
                        "height": height,
                        "image": image,
                        "num_results_per_prompt": 1,
                        "prompt": prompt,
                        "prompt_2": prompt_2,
                        "negative_prompt": negative_prompt,
                        "negative_prompt_2": negative_prompt_2,
                        "strength": strength,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "tiling_unet": tiling_unet,
                        "tiling_vae": tiling_vae,
                        "tiling_size": tiling_size,
                        "tiling_stride": tiling_stride,
                        "tiling_mask_type": tiling_mask_type,
                        "tiling_mask_kwargs": tiling_mask_kwargs,
                        "progress_callback": progress_callback,
                        "latent_callback": image_callback,
                        "latent_callback_type": "pil",
                        "latent_callback_steps": image_callback_steps,
                        "noise_offset": noise_offset,
                        "noise_method": noise_method,
                        "noise_blend_method": noise_blend_method,
                        "animation_frames": animation_frames,
                        "frame_window_size": frame_window_size,
                        "frame_window_stride": frame_window_stride,
                        "motion_scale": invocation_kwargs.get("motion_scale", None),
                        "tile": invocation_kwargs.get("tile", None),
                        "loop": invocation_kwargs.get("loop", False),
                    }

                    if controlnet:
                        logger.debug(f"Enabling controlnet {controlnet} for upscaling")

                        if refiner:
                            pipeline.refiner_controlnets = controlnet # type: ignore[assignment]
                            upscale_pipline = pipeline.refiner_pipeline
                            is_sdxl = pipeline.refiner_is_sdxl
                        elif animation_frames:
                            pipeline.animator_controlnets = controlnet # type: ignore[assignment]
                            upscale_pipeline = pipeline.animator_pipeline
                            is_sdxl = pipeline.animator_is_sdxl
                        else:
                            pipeline.controlnets = controlnet # type: ignore[assignment]
                            upscale_pipeline = pipeline.pipeline # type: ignore[assignment]
                            is_sdxl = pipeline.is_sdxl

                        with pipeline.control_image_processor.processor(controlnet) as process:
                            kwargs["control_images"] = dict([(controlnet, [(process(image), controlnet_scale)])])

                    elif refiner:
                        pipeline.refiner_controlnets = None # type: ignore[assignment]
                        upscale_pipeline = pipeline.refiner_pipeline # type: ignore[assignment]
                    elif animation_frames:
                        pipeline.animator_controlnets = None # type: ignore[assignment]
                        upscale_pipeline = pipeline.animator_pipeline
                    else:
                        pipeline.controlnets = None # type: ignore[assignment]
                        upscale_pipeline = pipeline.pipeline # type: ignore[assignment]

                    if upscale_pipeline.ip_adapter_loaded:
                        logger.debug(f"Upscale pipeline has IP adapter loaded, adding image to adapter input.")
                        kwargs["ip_adapter_images"] = [(image, 1.0)]

                    logger.debug(f"Upscaling sample {i} with arguments {kwargs}")
                    pipeline.stop_keepalive() # Stop here to kill during upscale diffusion
                    if task_callback:
                        task_callback(f"Re-diffusing Upscaled Sample {i+1}")

                    image = upscale_pipeline( # type: ignore[union-attr]
                        generator=pipeline.generator,
                        device=pipeline.device,
                        offload_models=pipeline.pipeline_sequential_onload,
                        **kwargs
                    ).images

                    if animation_frames:
                        images = image
                    else:
                        images[i] = image[0]
                        if image_callback is not None:
                            image_callback(images)

                if re_enable_safety:
                    pipeline.safe = True
                if refiner:
                    logger.debug("Offloading refiner for next inference.")
                    pipeline.refiner_controlnets = None # type: ignore[assignment]
                    pipeline.offload_refiner()
                elif animation_frames:
                    pipeline.animator_controlnets = None # type: ignore[assignment]
                else:
                    pipeline.controlnets = None # type: ignore[assignment]

        return images, nsfw

    def execute_interpolate(
        self,
        pipeline: DiffusionPipelineManager,
        images: List[Image],
        nsfw: List[bool],
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Finishes the result and interpolates
        """
        result = {
            "images": images,
            "nsfw_content_detected": nsfw
        }

        if not self.animation_frames or not (self.interpolate_frames or self.reflect) or (len(nsfw) > 0 and nsfw[0]):
            return result

        from enfugue.diffusion.util import interpolate_frames, reflect_frames
        with pipeline.interpolator.film() as interpolate:
            pipeline.stop_keepalive()
            if self.interpolate_frames:
                if task_callback is not None:
                    task_callback("Interpolating")
                if self.loop:
                    # Add copy of first frame to end
                    result["images"].append(result["images"][0])
                result["frames"] = [
                    frame for frame in interpolate_frames(
                        frames=result["images"],
                        multiplier=self.interpolate_frames,
                        interpolate=interpolate,
                        progress_callback=progress_callback
                    )
                ]
                if self.loop:
                    # remove copy of first frame from end
                    result["frames"] = result["frames"][:-1]
                    result["images"] = result["images"][:-1]
            else:
                result["frames"] = result["images"]
            if self.reflect:
                if task_callback is not None:
                    task_callback("Reflecting")
                result["frames"] = [
                    frame for frame in reflect_frames(
                        frames=result["frames"],
                        interpolate=interpolate,
                        progress_callback=progress_callback
                    )
                ]
        return result

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Gets the plan's metadata
        """
        metadata_dict = self.serialize()
        redact_images_from_metadata(metadata_dict)
        return metadata_dict
