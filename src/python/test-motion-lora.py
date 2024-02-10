# type: ignore
import os
from pibble.util.log import DebugUnifiedLoggingContext

from enfugue.diffusion.util import GridMaker, Video
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.train.motion.helper import MotionTrainer

here = os.path.abspath(os.getcwd())
enfugue_root = os.path.join(os.path.abspath(os.path.expanduser("~")), ".cache", "enfugue")
seed = 42
lora_name = "rotating_dolly_zoom_out_motion_mmv3_adapter_v1"
training_prompt = "A rotating dooly zoom out of a verdant landscape, blue lake, green grass, distant mountains"
validation_prompt = [
    training_prompt,
    "a rotating dolly zoom out of a man standing on a street corner",
    "a rotating dolly zoom out of a cabin in the woods during winter",
    "a rotating dolly zoom out of a woman wearing a red dress on stage"
]

validation_negative_prompt = "poor quality"

"""
validation_prompt = [
    "a suburban home on fire, burning brightly in the night",
    "a ripple in space-time, science fiction warping effect",
    "smoke rising from a chimney"
]
"""
num_prompts = len(validation_prompt)
quality = "best"
checkpoint_path = os.path.join(enfugue_root, "checkpoint", "epicrealism_pureEvolutionV5.safetensors")
# motion_module_path = os.path.join(enfugue_root, "motion", "animatediffMotion_v15.ckpt")
# motion_module_path = os.path.join(enfugue_root, "motion", "mm_sd_v15_v2.ckpt")
motion_module_path = os.path.join(enfugue_root, "motion", "v3_sd15_mm.ckpt")
# domain_adapter_path = None
domain_adapter_path = os.path.join(enfugue_root, "lora", "v3_sd15_adapter.ckpt")
lora_path = os.path.join(enfugue_root, "lora", "sd_v15_dpo_lora_v1.safetensors")

def do_training():
    trainer = MotionTrainer.simple(
        quality=quality,
        checkpoint_path=checkpoint_path,
        motion_module_path=motion_module_path,
        domain_adapter_path=domain_adapter_path,
        lora_path=lora_path,
        seed=seed,
    )
    samples = trainer(
        name=lora_name,
        cache_dir=os.path.join(enfugue_root, "cache"),
        output_dir=os.path.join(here, "output"),
        video_path=os.path.join(here, "input.mp4"),
        training_prompt=training_prompt,
        device="cuda",
        validation_prompt=validation_prompt,
        validation_negative_prompt=validation_negative_prompt,
        train_spatial=False
    )
    ordered_videos = []
    for i in range(num_prompts):
        ordered_videos += [
            video for j, video in enumerate(samples)
            if j % num_prompts == i
        ]

    grid_maker = GridMaker(
        grid_columns=len(samples)//num_prompts,
        use_video=True
    )
    grid = grid_maker.collage([
        ({}, f"Step {step}, \"{prompt}\"", images, None)
        for step, prompt, images in ordered_videos
    ])
    Video(grid).save(
        os.path.join("./training-results.mp4"),
        rate=8.0,
        overwrite=True
    )

def do_evaluation(model: str) -> None:
    lora = os.path.basename(model)
    manager = DiffusionPipelineManager()
    manager.model = checkpoint_path
    manager.motion_module = "v3_sd15_mm.ckpt"
    manager.lora = [lora_path, model, "v3_sd15_adapter.ckpt"]

    results = []

    def evaluate(prompt, label):
        nonlocal results
        manager.seed = seed
        result = manager(
            prompt=prompt,
            width=384,
            height=384,
            animation_frames=16,
            num_inference_steps=25,
            guidance_scale=8.5
        ).images
        Video(result).save("./tmp.gif", rate=8.0, overwrite=True)
        results.append(({}, f"{lora}, inference on {label}, {prompt}", result, None))

    for i, prompt in enumerate(validation_prompts):
        evaluate(prompt, "MMV3 w/ Adapter")

    manager.lora = [lora_path, model]

    for i, prompt in enumerate(validation_prompts):
        evaluate(prompt, "MMV3")

    manager.motion_module = "mm_sd_v15_v2.ckpt"

    for i, prompt in enumerate(validation_prompts):
        evaluate(prompt, "MMV2")

    manager.motion_module = "animatediffMotion_v15.ckpt"

    for i, prompt in enumerate(validation_prompts):
        evaluate(prompt, "MMV1")

    grid_maker = GridMaker(
        grid_columns=len(validation_prompts),
        use_video=True
    )
    grid = grid_maker.collage(results)
    Video(grid).save(
        os.path.join("./evaluation-results.mp4"),
        rate=8.0,
        overwrite=True
    )

with DebugUnifiedLoggingContext():
    do_training()
