import os
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import load_state_dict, Audio
from pibble.util.log import DebugUnifiedLoggingContext

with DebugUnifiedLoggingContext():
    manager = DiffusionPipelineManager()
    prompts = [
        "Hello! My name is Brea, and this is the default voice with no enhancement."
    ]
    with manager.audio.metavoice() as metavoice:
        samples = metavoice(
            texts=prompts,
            enhancer=None
#            embeddings=["./epic.pt"]*len(prompts),
        )
        Audio.combine(*samples, silence=1.0).save("./output.wav")
        prompts = [
            "Hello! My name is Brea, and this is the default voice with simple enhancement."
        ]
        samples = metavoice(texts=prompts)
        Audio.combine(*samples, silence=1.0).save("./output-enhance.wav")
