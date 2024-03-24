"""
Tests automatic loading of motion module/animator pipeline
"""
import os
import torch

from datetime import datetime

from typing import Literal, Optional

from enfugue.util import logger, fit_image, profiler, image_from_uri
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.constants import *
from enfugue.diffusion.util import Video, GridMaker

from PIL import Image

from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "i2v")
    input_dir = os.path.join(here, "test-images")
    input_image = image_from_uri(os.path.join(input_dir, "i2v.png"))
    input_image_2 = image_from_uri(os.path.join(input_dir, "i2v2.png"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        manager = DiffusionPipelineManager()
        manager.dynamicrafter_model = "1024"
        result = manager.dynamicrafter(
            [input_image],
            prompt="a woman puts on a record and dances",
            fs=12,
            num_frames=32,
            frame_window_size=16,
            frame_window_stride=8,
#            mask=image_from_uri(os.path.join(input_dir, "i2v-mask.png"))
        )
        Video(result).save(
            os.path.join(output_dir, "dynamicrafter.mp4"),
            overwrite=True,
            rate=8.0
        )
        del manager.dynamicrafter

if __name__ == "__main__":
    main()
