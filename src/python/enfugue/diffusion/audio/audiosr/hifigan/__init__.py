# type: ignore
# adapted from https://github.com/haoheliu/versatile_audio_super_resolution
from enfugue.diffusion.support.audio.audiosr.hifigan.models_v2 import Generator
from enfugue.diffusion.support.audio.audiosr.hifigan.models import Generator as Generator_old


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
