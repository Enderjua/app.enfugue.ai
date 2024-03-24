from __future__ import annotations

from typing import Dict, Type, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor, dtype as TorchDType

__all__ = [
    "TorchDataTypeConverter",
    "tensor_to_image",
    "image_to_tensor",
]

def tensor_to_image(
    latents: Tensor,
    lower: float=0.0,
    upper: float=1.0,
    correction: float=0.5
) -> Image:
    """
    Converts tensor to pixels using torchvision.
    """
    import torch
    from PIL import Image

    from enfugue.util import logger

    if len(latents.shape) == 4:
        latents = latents.squeeze(0)

    value_range = upper - lower
    value_range_correction = value_range - 1.0

    c, h, w = latents.shape
    c_copy = 3 - c
    image = latents.detach().cpu()
    if c_copy > 0:
        image = torch.cat(
            [image] + [image[c-1:]] * c_copy,
            dim=0
        )
    image = torch.clamp(image, lower, upper)
    image = (image + value_range_correction) / value_range
    image = (image * 255.0) + correction
    image = torch.clamp(image, 0, 255).to(torch.uint8)
    image = image.permute(1,2,0)

    return Image.fromarray(image.numpy())

def image_to_tensor(
    image: Image,
    lower: float=0.0,
    upper: float=1.0,
    include_batch: bool=True
) -> Tensor:
    """
    Converts PIL image to tensor
    """
    import torch
    import cv2
    import numpy as np
    value_range = upper - lower
    value_range_correction = (value_range - 1.0) / 2.0
    tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0
    tensor = tensor - value_range_correction
    tensor = tensor * value_range
    if include_batch:
        return tensor.unsqueeze(0)
    return tensor

class TorchDataTypeConverter:
    """
    This class converts between numpy and torch types.
    Also provides helper functions for converting from strings.
    """

    @classmethod
    def from_string(cls, torch_type: str) -> DType:
        """
        Converts a string to a torch DType.
        """
        import torch
        try:
            return {
                "complex128": torch.complex128,
                "cdouble": torch.complex128,
                "complex": torch.complex64,
                "complex64": torch.complex64,
                "cfloat": torch.complex64,
                "cfloat64": torch.complex64,
                "cf64": torch.complex64,
                "double": torch.float64,
                "float64": torch.float64,
                "fp64": torch.float64,
                "float": torch.float32,
                "full": torch.float32,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "fp8": torch.float8_e4m3fn,
                "float8": torch.float8_e4m3fn,
                "float8_e4m3": torch.float8_e4m3fn,
                "float8_e4m3fn": torch.float8_e4m3fn,
                "fp84": torch.float8_e4m3fn,
                "float8_e5m2": torch.float8_e5m2,
                "fp85": torch.float8_e5m2,
                "uint8": torch.uint8,
                "int8": torch.int8,
                "int16": torch.int16,
                "short": torch.int16,
                "int": torch.int32,
                "int32": torch.int32,
                "long": torch.int64,
                "int64": torch.int64,
                "bool": torch.bool,
                "bit": torch.bool,
                "1": torch.bool
            }[torch_type[6:] if torch_type.startswith("torch.") else torch_type]
        except KeyError:
            raise ValueError(f"Unknown torch type '{torch_type}'")

    @classmethod
    def from_torch(cls, torch_type: DType) -> Type:
        """
        Gets the numpy type from torch.
        :raises: KeyError When type is unknown.
        """
        import torch
        import numpy as np
        torch_to_numpy: Dict[torch.dtype, Type] = {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
            torch.bool: np.bool_,
        }
        return torch_to_numpy[torch_type]

    @classmethod
    def from_numpy(cls, numpy_type: Type) -> DType:
        """
        Gets the torch type from nump.
        :raises: KeyError When type is unknown.
        """
        import torch
        import numpy as np
        numpy_to_torch: Dict[Type, torch.dtype] = {
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
            np.bool_: torch.bool,
        }
        return numpy_to_torch[numpy_type]

    @classmethod
    def convert(cls, type_to_convert: Optional[Union[DType, Type, str]]) -> Optional[DType]:
        """
        Converts to a torch DType
        """
        import torch
        if type_to_convert is None:
            return None
        if isinstance(type_to_convert, torch.dtype):
            return type_to_convert
        if isinstance(type_to_convert, str):
            return cls.from_string(str(type_to_convert)) # Raises
        try:
            return cls.from_numpy(type_to_convert)
        except KeyError:
            return cls.from_string(str(type_to_convert)) # Raises
