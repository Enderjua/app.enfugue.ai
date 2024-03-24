# type: ignore
# adapted from https://github.com/Fanghua-Yu/SUPIR/
import torch
import copy
import random

from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate

from enfugue.diffusion.support.upscale.supir.sgm.models.diffusion import DiffusionEngine
from enfugue.diffusion.support.upscale.supir.sgm.util import instantiate_from_config
from enfugue.diffusion.support.upscale.supir.sgm.modules.distributions.distributions import DiagonalGaussianDistribution
from enfugue.diffusion.support.upscale.supir.utils.colorfix import wavelet_reconstruction, adaptive_instance_normalization
from enfugue.diffusion.support.upscale.supir.utils.tilevae import VAEHook
from enfugue.diffusion.util.torch_util.dtype_util import TorchDataTypeConverter
from enfugue.util import logger

class SUPIRModel(DiffusionEngine):
    def __init__(self, control_stage_config, ae_dtype='fp32', diffusion_dtype='fp32', p_p='', n_p='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        control_model = instantiate_from_config(control_stage_config)
        self.model.load_control_model(control_model)
        self.first_stage_model.denoise_encoder = copy.deepcopy(self.first_stage_model.encoder)
        self.sampler_config = kwargs['sampler_config']
        
        self.ae_dtype = TorchDataTypeConverter.convert(ae_dtype)
        self.model.dtype = TorchDataTypeConverter.convert(diffusion_dtype)

        self.p_p = p_p
        self.n_p = n_p

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def encode_first_stage_with_denoise(self, x, use_sample=True, is_stage1=False):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            if is_stage1:
                h = self.first_stage_model.denoise_encoder_s1(x)
            else:
                h = self.first_stage_model.denoise_encoder(x)
            moments = self.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            if use_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", dtype=self.ae_dtype):
            out = self.first_stage_model.decode(z)
        return out.float()

    @torch.no_grad()
    def batchify_denoise(self, x, is_stage1=False):
        '''
        [N, C, H, W], [-1, 1], RGB
        '''
        x = self.encode_first_stage_with_denoise(x, use_sample=False, is_stage1=is_stage1)
        return self.decode_first_stage(x)

    @torch.no_grad()
    def batchify_sample(
        self,
        x,
        p,
        p_p='default',
        n_p='default',
        num_steps=100,
        restoration_scale=4.0,
        s_churn=0,
        s_noise=1.003,
        cfg_scale=4.0,
        seed=-1,
        num_samples=1,
        control_scale=1,
        color_fix_type='None',
        use_linear_CFG=False,
        use_linear_control_scale=False,
        cfg_scale_start=1.0,
        control_scale_start=0.0,
        sampler_callback=None,
        **kwargs
    ):
        '''
        [N, C], [-1, 1], RGB
        '''
        assert len(x) == len(p)
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']

        N = len(x)
        if num_samples > 1:
            assert N == 1
            N = num_samples
            x = x.repeat(N, 1, 1, 1)
            p = p * N

        if p_p == 'default':
            p_p = self.p_p
        if n_p == 'default':
            n_p = self.n_p

        self.sampler_config.params.num_steps = num_steps
        if use_linear_CFG:
            self.sampler_config.params.guider_config.params.scale_min = cfg_scale
            self.sampler_config.params.guider_config.params.scale = cfg_scale_start
        else:
            self.sampler_config.params.guider_config.params.scale_min = cfg_scale
            self.sampler_config.params.guider_config.params.scale = cfg_scale
        self.sampler_config.params.restore_cfg = restoration_scale
        self.sampler_config.params.s_churn = s_churn
        self.sampler_config.params.s_noise = s_noise
        logger.debug(f"Instantiating sampler from configuration {self.sampler_config}")
        self.sampler = instantiate_from_config(self.sampler_config)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z_stage1 = self.encode_first_stage(x_stage1)

        c, uc = self.prepare_condition(_z, p, p_p, n_p, N)

        denoiser = lambda input, sigma, c, control_scale: self.denoiser(
            self.model, input, sigma, c, control_scale, **kwargs
        )

        noised_z = torch.randn_like(_z).to(_z.device)

        _samples = self.sampler(denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1, control_scale=control_scale, use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start, callback=sampler_callback)
        samples = self.decode_first_stage(_samples)
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)
        return samples

    def init_tile_vae(self, encoder_tile_size=512, decoder_tile_size=64, device="cpu"):
        self.first_stage_model.denoise_encoder.original_forward = self.first_stage_model.denoise_encoder.forward
        self.first_stage_model.encoder.original_forward = self.first_stage_model.encoder.forward
        self.first_stage_model.decoder.original_forward = self.first_stage_model.decoder.forward
        self.first_stage_model.denoise_encoder.forward = VAEHook(
            self.first_stage_model.denoise_encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True, device=device)
        self.first_stage_model.encoder.forward = VAEHook(
            self.first_stage_model.encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True, device=device)
        self.first_stage_model.decoder.forward = VAEHook(
            self.first_stage_model.decoder, decoder_tile_size, is_decoder=True, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True, device=device)

    def prepare_condition(self, _z, p, p_p, n_p, N):
        batch = {}
        batch['original_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['crop_coords_top_left'] = torch.tensor([0, 0]).repeat(N, 1).to(_z.device)
        batch['target_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['aesthetic_score'] = torch.tensor([9.0]).repeat(N, 1).to(_z.device)
        batch['control'] = _z

        batch_uc = copy.deepcopy(batch)
        batch_uc['txt'] = [n_p for _ in p]

        if not isinstance(p[0], list):
            batch['txt'] = [''.join([_p, p_p]) for _p in p]
            with torch.cuda.amp.autocast(dtype=self.ae_dtype):
                c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
        else:
            assert len(p) == 1, 'Support bs=1 only for local prompt conditioning.'
            p_tiles = p[0]
            c = []
            for i, p_tile in enumerate(p_tiles):
                batch['txt'] = [''.join([p_tile, p_p])]
                with torch.cuda.amp.autocast(dtype=self.ae_dtype):
                    if i == 0:
                        _c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
                    else:
                        _c, _ = self.conditioner.get_unconditional_conditioning(batch, None)
                c.append(_c)
        return c, uc
