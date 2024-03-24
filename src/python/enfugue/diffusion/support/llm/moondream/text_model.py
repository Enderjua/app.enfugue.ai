# type: ignore
# adapted from https://huggingface.co/vikhyatk/moondream1/
from torch import nn
import transformers

from enfugue.diffusion.support.llm.moondream.modeling_phi import PhiForCausalLM
from enfugue.diffusion.support.llm.moondream.configuration_moondream import PhiConfig

transformers.logging.set_verbosity_error()


class TextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if type(config.phi_config) == dict:
            phi_config = PhiConfig(**config.phi_config)
        else:
            phi_config = config.phi_config

        self.model = PhiForCausalLM(phi_config)
        self.text_emb = self.model.get_input_embeddings()
