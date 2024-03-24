from __future__ import annotations

from typing import List, Union, Iterator, TYPE_CHECKING
from contextlib import contextmanager

from enfugue.diffusion.support.model import SupportModel, SupportModelProcessor

if TYPE_CHECKING:
    from PIL.Image import Image


__all__ = ["SegmentationDetector"]

class SegmentationImageProcessor(SupportModelProcessor):
    """
    Holds a reference to the segment anything model.
    """
    def __init__(self, generator: SamAutomaticMaskGenerator) -> None:
        self.generator = generator

    def __call__(
        self,
        image: Union[str, Image],
        output_type: Literal["binary-mask", "mask", "slice", "cropped-slice"]="mask"
    ) ->List[Image]:
        """
        Calls the generator
        """
        from PIL import Image
        from enfugue.diffusion.util import ComputerVision

        if isinstance(image, str):
            image = Image.open(image)

        results = self.generator.generate(ComputerVision.convert_image(image))

        if output_type == "binary-mask":
            return [result["segmentation"] for result in results]

        masks = [
            Image.fromarray(result["segmentation"])
            for result in results
        ]
        if output_type == "mask":
            return masks

        images: List[Image.Image] = []
        for mask in masks:
            blank = Image.new("RGBA", image.size)
            blank.paste(image, mask=mask)
            images.append(blank)
        if output_type == "slice":
            return images

        return [
            image.crop(image.getbbox())
            for image in images
        ]

class SegmentationDetector(SupportModel):
    """
    Used to separate images into their constituent segments.
    """
    SAM_MODEL_PATH = "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth"

    @property
    def segment_anything_checkpoint(self) -> str:
        """
        Gets the SAM checkpoint.
        """
        return self.get_model_file(self.SAM_MODEL_PATH)

    @contextmanager
    def sam(self, crop_n_layers: int=0) -> Iterator[SegmentationImageProcessor]:
        """
        Gets the segment anything model.
        """
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=self.segment_anything_checkpoint)
        sam = sam.to(device=self.device)
        generator = SamAutomaticMaskGenerator(
            sam,
            crop_n_layers=1,
            min_mask_region_area=1e4
        )
        processor = SegmentationImageProcessor(generator)
        yield processor
        del processor
        del generator
