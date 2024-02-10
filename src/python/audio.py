# type: ignore
import os
import gc
import scipy
import torch
import torch.cuda
import numpy as np

from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image
from diffusers import AudioLDM2Pipeline
from random import random
from audiosr import super_resolution, build_model, save_wave, get_time, read_list
from enfugue.diffusion.util import Video
from pibble.util.log import DebugUnifiedLoggingContext

audio_length = 10.0
num_inference_steps = 200
num_sr_inference_steps = 200
guidance_scale = 4.0
num_waveforms = 3
rate = 1.0
audio_rate = 16000
sr_audio_rate = 48000

height_chunks = 2
width_chunks = 2

image = Image.open("./iso.png")

with DebugUnifiedLoggingContext():
    """
    model_id = "vikhyatk/moondream1"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to("cuda")
    tokenizer = Tokenizer.from_pretrained(model_id)

    width, height = image.size
    x_slices = np.linspace(0, width, width_chunks + 1).astype(np.uint16)
    y_slices = np.linspace(0, height, height_chunks + 1).astype(np.uint16)

    images = [image]
    for i in range(width_chunks):
        width_start, width_end = x_slices[i:i+2]
        for j in range(height_chunks):
            height_start, height_end = y_slices[j:j+2]
            images.append(
                image.crop(
                    (width_start, height_start, width_end, height_end)
                )
            )

    prompts = []
    for im in images:
        enc_image = model.encode_image(im)

        query = "Precisely describe the sounds that could be heard by a listener in the image using one sentence." # My prompt

        prompt = model.answer_question(enc_image, query, tokenizer)

        prompts.append(f"sound effect, {prompt}, high quality")
        print(prompt)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    """
    prompts = ["sound effect, a chainsaw cutting down a tree"]
    repo_id = "cvssp/audioldm2-large"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    audio = pipe(
        num_waveforms_per_prompt=num_waveforms,
        prompt=prompts,
        negative_prompt=["low quality,worst quality"]*len(prompts),
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length,
        guidance_scale=guidance_scale,
    ).audios

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    num_sounds = len(audio)
    final_audio = None
    for i, sound in enumerate(audio):
        # Allow between -0.5 and 0.5, with 0.0 being center
        stereo_ratio_start = random() - 0.5
        stereo_ratio_end = random() - 0.5
        stereo_ratio = np.linspace(stereo_ratio_start, stereo_ratio_end, len(sound))
        left_mult = np.where(stereo_ratio < 0, 1.0, 1.0-stereo_ratio)
        right_mult = np.where(stereo_ratio > 0, 1.0, stereo_ratio+1.0)
        total_mult = 0.8 - ((i / (num_sounds - 1)) * 0.6)
        sound = np.array([
            sound * left_mult * total_mult,
            sound * right_mult * total_mult
        ])
        if final_audio is None:
            final_audio = sound
        else:
            final_audio += sound

    scipy.io.wavfile.write("./output.wav", rate=audio_rate, data=final_audio.T)
    audiosr = build_model(model_name="basic", device="cuda")
    waveform = super_resolution(
        audiosr,
        "./output.wav",
        seed=42,
        guidance_scale=3.5,
        ddim_steps=num_sr_inference_steps,
        latent_t_per_second=12.8
    )
    save_wave(
        waveform,
        inputpath="./output.wav",
        savepath=os.path.dirname(os.path.abspath(__file__)),
        name="output-upsampled",
        samplerate=sr_audio_rate
    )
    video = Video(
        frames=[image] * int(rate*audio_length),
        frame_rate=rate,
        audio_rate=sr_audio_rate,
        audio="./output-upsampled.wav",
    )
    video.save("./output.mp4", rate=rate, overwrite=True)
