# type: ignore
import os
import numpy as np

from enfugue.diffusion.util import Audio
from audiosr import super_resolution, build_model, save_wave, get_time, read_list

audiosr = build_model(model_name="basic", device="cuda")

for filename in os.listdir("./keep"):
    basename, ext = os.path.splitext(filename)
    waveform = list(Audio.file_to_frames(f"./keep/{filename}"))
    frames = len(waveform)
    chunks = np.ceil(frames/92160).astype(np.uint32)
    chunks = np.linspace(0, len(waveform), chunks).astype(np.uint32)
    upsampled_waveform = []
    for i in range(len(chunks)-1):
        sub_waveform = waveform[chunks[i]:chunks[i+1]]
        temp_file = f"./keep/{basename}-{i}.wav"
        temp_upsample_file = f"./keep/{basename}-{i}-upsample.wav"
        Audio(sub_waveform).save(temp_file)
        sr_waveform = super_resolution(
            audiosr,
            temp_file,
            seed=42,
            guidance_scale=3.5,
            ddim_steps=50,
            latent_t_per_second=12.8
        )
        save_wave(
            sr_waveform,
            inputpath=temp_file,
            savepath="./keep",
            name=f"{basename}-{i}-upsample",
            samplerate=48000
        )
        upsampled_waveform += list(Audio.file_to_frames(temp_upsample_file))
        os.remove(temp_upsample_file)
        os.remove(temp_file)
    Audio(upsampled_waveform).save(f"./keep/{basename}-upsampled.wav")
