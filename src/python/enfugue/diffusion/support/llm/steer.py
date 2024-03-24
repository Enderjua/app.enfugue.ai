# Adapted from https://github.com/Mihaiii/llm_steer
# MIT License
from __future__ import annotations

from copy import deepcopy
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import List, Any, TYPE_CHECKING
from enum import Enum

from enfugue.util.log import logger

if TYPE_CHECKING:
    from torch import Tensor, torch

class ActivationMode(Enum):
    ORIGINAL = 1
    CAPTURE = 2
    STEER = 3


@dataclass
class SteerElement:
    text: str
    tensor: Tensor
    coeff: float
    try_keep_nr: int
    exclude_bos_token: bool = False
    
@dataclass
class SteerData:
    orig_forward_fn: torch.nn.Module.forward
    layer_idx: int
    steer_vectors: List[SteerElement]


class Steer:
    steers = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        copy_model: bool = False,
    ) -> None:
        import torch
        self.model = deepcopy(model) if copy_model else model
        self.tokenizer = tokenizer
        self.device = torch.device(next(model.parameters()).device)

    @property
    def num_layers(self) -> int:
        return len(self.model._modules["model"].layers)

    def _set_forward_fn(self, option: ActivationMode, layer_idx: int) -> None:
        if option == ActivationMode.ORIGINAL:
            steer = self.steers.pop(layer_idx, None)
            if steer is not None:
                self.model._modules["model"].layers[
                    layer_idx
                ].forward = steer.orig_forward_fn
        elif option == ActivationMode.CAPTURE:
            self.steers.setdefault(
                layer_idx,
                SteerData(
                    orig_forward_fn=self.model._modules["model"]
                    .layers[layer_idx]
                    .forward,
                    layer_idx=layer_idx,
                    steer_vectors=[],
                ),
            )
            self.model._modules["model"].layers[
                layer_idx
            ].forward = self._store_activations_forward(layer_idx)
        elif option == ActivationMode.STEER:
            self.steers.setdefault(
                layer_idx,
                SteerData(
                    orig_forward_fn=self.model._modules["model"]
                    .layers[layer_idx]
                    .forward,
                    layer_idx=layer_idx,
                    steer_vectors=[],
                ),
            )
            self.model._modules["model"].layers[
                layer_idx
            ].forward = self._steer_vector_forward(layer_idx)

    def _store_activations_forward(self, layer_idx: int) -> None:
        def _store_activations_forward_inner(*args: Any, **kwargds: Any) -> None:
            self.captured_tensor = (
                kwargds["hidden_states"] if "hidden_states" in kwargds else args[0]
            )
            return self.steers[layer_idx].orig_forward_fn(*args, **kwargds)

        return _store_activations_forward_inner

    def _steer_vector_forward(self, layer_idx: int) -> None:
        import torch
        def _steer_vector_forward_inner(*args: Any, **kwargds: Any) -> None:
            for elem in self.steers[layer_idx].steer_vectors:
                if elem.tensor.size()[1] <= elem.try_keep_nr:
                    extra_text = ""
                    if elem.tensor.size()[1] == elem.try_keep_nr:
                        extra_text = """ In case you're using exclude_bos_token=True, 
                        you could also consider setting it to False and retrying."""
                    raise Exception(
                        f"""Invalid try_keep_nr value. Current value is {elem.try_keep_nr}, 
                    but it has to be less than {elem.tensor.size()[1]} on layer index {layer_idx}. 
                    You could set a lower value for try_keep_nr or provide longer text for steering. {extra_text}"""
                    )

                delta = torch.mean(
                    elem.coeff * elem.tensor[:, elem.try_keep_nr :, :],
                    dim=1,
                    keepdim=True,
                )

                if "hidden_states" in kwargds:
                    # if user generates with use_cache=True, we receive only the latest row
                    # otherwise (use_cache=False), we receive the whole matrix, that will keep expanding
                    if kwargds["hidden_states"].size()[1] == 1:
                        kwargds["hidden_states"][:, -1:, :] += delta
                    else:
                        kwargds["hidden_states"][:, elem.try_keep_nr :, :] += delta
                elif isinstance(args[0], torch.Tensor):
                    if args[0].size()[1] == 1:
                        args[0][:, -1:, :] += delta
                    else:
                        args[0][:, elem.try_keep_nr :, :] += delta
                else:
                    raise Exception(
                        "The model is not currently supported. Plase open an issue in the official github repository."
                    )

            return self.steers[layer_idx].orig_forward_fn(*args, **kwargds)

        return _steer_vector_forward_inner

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all the steering vectors data that are applied on the model.
        Can be used for replicating in the future the state.
        """
        return [{
            'layer_idx': val.layer_idx,
            'text': x.text,
            'coeff': x.coeff,
            'try_keep_nr': x.try_keep_nr,
            'exclude_bos_token': x.exclude_bos_token
        } for val in self.steers.values() for x in val.steer_vectors]

    def reset(self, layer_idx: int) -> None:
        """
        Remove the steering vectors on a particular layer.      

        Args:
            layer_idx (int): The layer index that will have the steering vectors removed.
        """
        self._set_forward_fn(ActivationMode.ORIGINAL, layer_idx)

    def reset_all(self) -> None:
        """
        Remove all steering vectors that were applied on the model.
        Gets the model to initial state, before wrapping it in the Steer class and using add(). 
        """
        [self.reset(idx) for idx in range(len(self.model._modules["model"].layers))]

    def add(
        self,
        layer_idx: int,
        coeff: float,
        text: str,
        try_keep_nr: int = None,
        exclude_bos_token: bool = False,
    ) -> None:
        """
        Add a steering vector.
        Args:
            layer_idx (int): The layer index to apply the steering vector on. Usually is toward the end.
            coeff: The steerging vectors coefficient. Usually is below 1. Can also be negative.
            text: The steering vector text.
            try_keep_nr: This is used in advanced usage and determines the number of rows of the initial
                matrix to be kept. The param is used for expetimenting. Leave to default value for best usage.
            exclude_bos_token: This is used in advanced usage and determines if the beginning of a sentence
                (bos) token should be removed. By default, the code ensures the tokens used for generating
                start with the bos token. The param is used for expetimenting. Leave to default value for best usage.
        """
        
        assert layer_idx >= 0 and layer_idx < self.num_layers, f"Layer must be in the range 0 <= index < {self.num_layers}"

        if layer_idx in self.steers:
            self.reset(layer_idx)

        import torch

        text_tokens = self.tokenizer.encode(text)

        #inject bos_token
        #This can be reverted with exclude_bos_token=True
        if self.tokenizer.bos_token is not None and text_tokens[0] != self.tokenizer.encode(self.tokenizer.bos_token)[-1]:
            text_tokens.insert(0, self.tokenizer.encode(self.tokenizer.bos_token)[-1])
        
        if (
            exclude_bos_token
            and self.tokenizer.bos_token is not None
        ):
            text_tokens = text_tokens[1:]
        
        layer_tensor = self._capture_tensor(
            layer_idx, torch.tensor(text_tokens).to(self.device).unsqueeze(0)
        )

        if try_keep_nr is None:
            try_keep_nr = 0 if self.tokenizer.bos_token is None else 1

        self._add_steer_vector(
            layer_idx,
            SteerElement(
                text=text, tensor=layer_tensor, coeff=coeff, try_keep_nr=try_keep_nr, exclude_bos_token=exclude_bos_token
            ),
        )

    def add_all(
        self,
        coeff: float,
        text: str,
        try_keep_nr: int = None,
        exclude_bos_token: bool = False,
    ) -> None:
        """
        Adds a steering vector to all layers.
        """
        import numpy as np
        num_layers = self.num_layers
        midpoint = (num_layers - 1) / 2
        growth = 0.1
        deviation = 0.01
        coefficients = np.array([
            np.exp(-(x-midpoint)*(x-midpoint)/(num_layers**(2+growth))/(2*deviation))/np.sqrt(2*np.pi*deviation)
            for x in range(num_layers)
        ])
        coefficients /= max(coefficients)
        for i, layer_coeff in enumerate(coefficients):
            self.add(
                layer_idx=i,
                coeff=coeff*layer_coeff*((num_layers-i)/num_layers),
                text=text,
                try_keep_nr=try_keep_nr,
                exclude_bos_token=exclude_bos_token
            )

    def _add_steer_vector(self, layer_idx: int, steer_element: SteerElement) -> None:
        steer = self.steers.setdefault(
            layer_idx,
            SteerData(
                orig_forward_fn=self.model._modules["model"].layers[layer_idx].forward,
                layer_idx=layer_idx,
                steer_vectors=[],
            ),
        )
        steer.steer_vectors.append(steer_element)
        self._set_forward_fn(ActivationMode.STEER, layer_idx)

    def _capture_tensor(self, layer_idx: int, tokens: Tensor) -> None:
        self._set_forward_fn(ActivationMode.CAPTURE, layer_idx)
        self.model(tokens)
        result = self.captured_tensor
        return result

    def __str__(self) -> str:
        return "(empty steering)" if not self.steers else ", ".join([
            f"layer {d['layer_idx']}: {d['coeff']:0.4f}({d['text']})"
            for d in self.get_all()
        ])
