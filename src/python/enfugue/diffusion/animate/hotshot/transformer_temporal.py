# type: ignore
# Copyright 2023 Natural Synthetics Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import torch
import math
from dataclasses import dataclass
from torch import nn
from diffusers.utils import BaseOutput
from diffusers.models.attention import Attention, FeedForward
from einops import rearrange, repeat
from typing import Optional

def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start,t_end))
    return views

def generate_weight_sequence(n):
    if n % 2 == 0:
        max_weight = n // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (n + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention Is All You Need".
    Adds sinusoidal based positional encodings to the input tensor.
    """

    _SCALE_FACTOR = 10000.0  # Scale factor used in the positional encoding computation.

    def __init__(self, dim: int, dropout: float = 0.0, max_length: int = 24):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # The size is (1, max_length, dim) to allow easy addition to input tensors.
        positional_encoding = torch.zeros(1, max_length, dim)

        # Position and dim are used in the sinusoidal computation.
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(self._SCALE_FACTOR) / dim))

        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)

        # Register the positional encoding matrix as a buffer,
        # so it's part of the model's state but not the parameters.
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, hidden_states: torch.Tensor, length: int, scale: float = 1.0) -> torch.Tensor:
        hidden_states = hidden_states + self.positional_encoding[:, :length] * (1.0 + scale)
        return self.dropout(hidden_states)


class TemporalAttention(Attention):
    def __init__(
        self,
        positional_encoding_max_length: int = 24,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.is_cross_attention = kwargs.get("cross_attention_dim", None) is not None
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0,
            max_length=positional_encoding_max_length
        )

    def set_scale_multiplier(self, multiplier: float = 1.0) -> None:
        if not hasattr(self, "_default_scale"):
            self._default_scale = self.scale
        self.scale = multiplier / (self.inner_dim // self.heads)

    def reset_scale_multiplier(self) -> None:
        self.scale = self._default_scale

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, number_of_frames=8):
        sequence_length = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) s c -> (b s) f c", f=number_of_frames)
        hidden_states = self.pos_encoder(hidden_states, length=number_of_frames, scale=self.scale)

        if encoder_hidden_states is not None:
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b s) n c", s=sequence_length)

        hidden_states = super().forward(hidden_states, encoder_hidden_states, attention_mask=attention_mask)

        return rearrange(hidden_states, "(b s) f c -> (b f) s c", s=sequence_length)


@dataclass
class TransformerTemporalOutput(BaseOutput):
    sample: torch.FloatTensor


class TransformerTemporal(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        positional_encoding_max_length: int = 24,
        upcast_attention: bool = False,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_attention_dim=cross_attention_dim,
                    positional_encoding_max_length=positional_encoding_max_length
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.is_cross_attention = bool(cross_attention_dim)

    def set_attention_scale_multiplier(self, attention_scale: float = 1.0) -> None:
        for block in self.transformer_blocks:
            block.set_attention_scale_multiplier(attention_scale)

    def reset_attention_scale_multiplier(self) -> None:
        for block in self.transformer_blocks:
            block.reset_attention_scale_multiplier()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        frame_window_size=None,
        frame_window_stride=None,
    ):
        _, num_channels, f, height, width = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        skip = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = rearrange(hidden_states, "bf c h w -> bf (h w) c")
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                number_of_frames=f,
                frame_window_size=frame_window_size,
                frame_window_stride=frame_window_stride,
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, "bf (h w) c -> bf c h w", h=height, w=width).contiguous()

        output = hidden_states + skip
        output = rearrange(output, "(b f) c h w -> b c f h w", f=f)

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        dropout=0.0,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        depth=2,
        positional_encoding_max_length=24,
        cross_attention_dim: Optional[int] = None
    ):
        super().__init__()

        self.is_cross = cross_attention_dim is not None

        attention_blocks = []
        norms = []

        for _ in range(depth):
            attention_blocks.append(
                TemporalAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    positional_encoding_max_length=positional_encoding_max_length
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def set_attention_scale_multiplier(self, attention_scale: float = 1.0) -> None:
        for block in self.attention_blocks:
            block.set_scale_multiplier(attention_scale)

    def reset_attention_scale_multiplier(self) -> None:
        for block in self.attention_blocks:
            block.reset_scale_multiplier()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        number_of_frames=None,
        frame_window_size=None,
        frame_window_stride=None,
    ):
        if not self.is_cross:
            encoder_hidden_states = None

        if frame_window_size and frame_window_stride:
            views = get_views(number_of_frames, frame_window_size, frame_window_stride)
            hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=number_of_frames)
            count = torch.zeros_like(hidden_states)
            value = torch.zeros_like(hidden_states)

            for t_start, t_end in views:
                weight_sequence = generate_weight_sequence(t_end - t_start)
                weight_tensor = torch.ones_like(count[:, t_start:t_end])
                weight_tensor = weight_tensor * torch.Tensor(weight_sequence).to(hidden_states.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                sub_hidden_states = rearrange(hidden_states[:, t_start:t_end], "b f d c -> (b f) d c")
                for attention_block, norm in zip(self.attention_blocks, self.norms):
                    norm_hidden_states = norm(sub_hidden_states)
                    sub_hidden_states = attention_block(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                        number_of_frames=t_end-t_start,
                    ) + sub_hidden_states

                sub_hidden_states = rearrange(sub_hidden_states, "(b f) d c -> b f d c", f=t_end-t_start)
                value[:,t_start:t_end] += sub_hidden_states * weight_tensor
                count[:,t_start:t_end] += weight_tensor

            hidden_states = torch.where(count>0, value/count, value)
            hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c") 
        else:
            for block, norm in zip(self.attention_blocks, self.norms):
                norm_hidden_states = norm(hidden_states)
                hidden_states = block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    number_of_frames=number_of_frames,
                ) + hidden_states

        norm_hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states

        output = hidden_states
        return output
