# type: ignore
import os
import gc
import numpy as np

from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

from enfugue.diffusion.util import Audio

model = MAGNeT.get_pretrained("facebook/magnet-medium-10secs")
model.set_generation_params(decoding_steps = [60, 10, 10, 10], span_arrangement = "stride1")

descriptions = [
    "heavy metal, industrial, power chord, guitar solo"
] * 10

qualifiers = [
    "high-quality",
    "energetic",
    "catchy",
    "award-winning",
    "powerful",
    "popular",
    "fast-paced",
    "high-tempo",
    "pounding",
    "frenzied"
]

descriptions = [
    "energetic EDM, sawtooth wave, high quality",
    "fast-paced electronic dance beat, powerful bassline, catchy synth melody",
    "Electronic dance music with a thumping beat, pulsing bass, and uplifting synths",
    "Catchy electronic dance tune, rhythm, driving bass, and soaring dance melodies",
    "Pumping electronic dance music with a pounding beat, throbbing bass, and hypnotic synths",
    "electronic dance music, fast-paced, high-energy, thumping beat, pulsing bass, and soaring synths,",
    "popular electronic dance music, beat, with a driving bassline, pulsing rhythms, catchy melodies",
    "frenzied electronic dance music, beat, with a pounding bassline, relentless rhythms, infectious melodies",
    "fast-paced electronic dance music, driving beat, pounding bass, soaring melodies",
    "downtempo EDM, dubstep, high quality"
]

for i in range(0, len(descriptions), 2):
    wav = model.generate(descriptions[i:i+2])  # generates 2 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx+i}', one_wav.cpu(), model.sample_rate, strategy="loudness")

#import sys
#sys.exit(0)

del wav
gc.collect()
import torch
import torch.cuda
torch.cuda.empty_cache()

from audiosr import super_resolution, build_model, save_wave, get_time, read_list
audiosr = build_model(model_name="basic", device="cuda")

for i in range(len(descriptions)):
    sr_waveform = super_resolution(
        audiosr,
        f"./{i}.wav",
        seed=42,
        guidance_scale=3.5,
        ddim_steps=50,
        latent_t_per_second=12.8
    )
    save_wave(
        sr_waveform,
        inputpath=f"./{i}.wav",
        savepath=os.path.dirname(os.path.abspath(__file__)),
        name=f"{i}-upsample",
        samplerate=48000
    )
