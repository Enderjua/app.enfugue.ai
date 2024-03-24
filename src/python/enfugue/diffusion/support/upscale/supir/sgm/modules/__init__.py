# type: ignore
# adapted from https://github.com/Fanghua-Yu/SUPIR/
from enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules import GeneralConditioner
from enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules import GeneralConditionerWithControl
from enfugue.diffusion.support.upscale.supir.sgm.modules.encoders.modules import PreparedConditioner

UNCONDITIONAL_CONFIG = {
    "target": "enfugue.diffusion.support.upscale.supir.sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
