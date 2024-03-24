# type: ignore
# adapted from https://github.com/hzwer/Practical-RIFE/
import torch
import torch.nn as nn

backwarp_grid_tensor = {}

def warp(input_tensor, flow_tensor):
    device = flow_tensor.device
    k = (str(device), str(flow_tensor.size()))
    if k not in backwarp_grid_tensor:
        horizontal_tensor = torch.linspace(
            -1.0, 1.0, flow_tensor.shape[3], device=device
        ).view(
            1, 1, 1, flow_tensor.shape[3]
        ).expand(
            flow_tensor.shape[0], -1, flow_tensor.shape[2], -1
        )

        vertical_tensor = torch.linspace(
            -1.0, 1.0, flow_tensor.shape[2], device=device
        ).view(
            1, 1, flow_tensor.shape[2], 1
        ).expand(
            flow_tensor.shape[0], -1, -1, flow_tensor.shape[3]
        )

        backwarp_grid_tensor[k] = torch.cat([
            horizontal_tensor,
            vertical_tensor
        ], dim=1).to(device)

    flow_tensor = torch.cat([
        flow_tensor[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
        flow_tensor[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)
    ], dim=1)

    grid = (backwarp_grid_tensor[k] + flow_tensor).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=input_tensor,
        grid=grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
