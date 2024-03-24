# type: ignore
# adapted from https://github.com/hzwer/Practical-RIFE/
import torch
import torch.nn as nn
import itertools
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import AdamW
from accelerate.utils import set_module_tensor_to_device

from enfugue.diffusion.util import iterate_state_dict
from enfugue.diffusion.support.interpolate.rife.IFNet_HDv3 import IFNet
from enfugue.diffusion.support.interpolate.rife.model.loss import EPE, SOBEL
from enfugue.diffusion.support.interpolate.rife.model.warplayer import warp

class Model:
    def __init__(self, path=None, device=None):
        self.flownet = IFNet()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 4.8
        self.sobel = SOBEL()
        if path is not None:
            self.load_model(path, device)
        if device is not None:
            self.to(device)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def to(self, device):
        self.flownet.to(device)
        self.sobel.to(device)
        self.epe.to(device)

    def load_model(self, path, device=None):
        device = torch.device("cpu") if device is None else device
        for key, value in iterate_state_dict(path):
            set_module_tensor_to_device(
                self.flownet,
                key.replace("module.", ""),
                device,
                value=value,
            )

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[3] - gt).abs().mean()
        loss_smooth = self.sobel(flow[3], flow[3]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[3], {
            'mask': mask,
            'flow': flow[3][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }
