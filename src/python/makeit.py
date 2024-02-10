import torch
import safetensors
import safetensors.torch
from pibble.util.files import dump_yaml
from collections import OrderedDict

tensors = {}
config = {}

config["first_stage"] = {}
config["second_stage"] = {}

first_stage = torch.load("./first_stage.pt", map_location="cpu")
second_stage = torch.load("./second_stage.pt", map_location="cpu")

def recurse_dict(state_dict, stage, prefix=None):
    for key in state_dict:
        full_key = key if prefix is None else f"{prefix}.{key}"
        full_key = full_key.replace("_orig_mod", "first_stage")
        if isinstance(state_dict[key], dict) or isinstance(state_dict[key], OrderedDict):
            recurse_dict(state_dict[key], stage, prefix=full_key)
        elif isinstance(state_dict[key], torch.Tensor):
            tensors[full_key] = state_dict[key]
        else:
            config[stage][key] = state_dict[key]

recurse_dict(first_stage, "first_stage")
recurse_dict(second_stage, "second_stage", "second_stage")

print(config)

safetensors.torch.save_file(tensors, "./metavoid-1B-v0.1.safetensors")
dump_yaml("./config.yaml", config)
