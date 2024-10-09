# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import annotations

import abc
import dataclasses
import logging
import math
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import torch
import transformers.modeling_outputs
from torch import Size, Tensor, nn
from transformers import (
    GPT2Config,
    PreTrainedModel,
)
from typedtensor import Dimension, Sub, TypedTensor
from typedtensor import pytorch as ttorch
from typedtensor.pytorch import nn as tnn

logger = logging.getLogger(__name__)

_CHECKPOINT_FOR_DOC = "openai-community/gpt2"

BATCH_SIZE = 2
NUM_HEADS = 3
HEAD_FEATURES_LENGTH = 10  # FEATURES_LENGTH // NUM_HEADS
FEATURES_LENGTH = NUM_HEADS * HEAD_FEATURES_LENGTH
VOCAB_SIZE = 6


# fmt: off
class BatchDim(Dimension, length=BATCH_SIZE): pass #noqa E701
class _SequenceDim(Dimension): pass #noqa E701
class PastSequenceDim(_SequenceDim): pass #noqa E701
class SequenceDim(PastSequenceDim): pass #noqa E701
class HeadDim(Dimension): pass #noqa E701
class FeatureDim(Dimension, length=FEATURES_LENGTH): pass #noqa E701
class HeadFeatureDim(Dimension, length=HEAD_FEATURES_LENGTH): pass #noqa E701
class DoubleFeatureDim(Dimension, length=2 * FEATURES_LENGTH): pass #noqa E701
class TribbleFeatureDim(Dimension, length=3 * FEATURES_LENGTH): pass #noqa E701
class VocabDim(Dimension, length=VOCAB_SIZE): pass #noqa E701


type HiddenStatesTypedTensor[DType: Tensor] = TypedTensor[DType, BatchDim, SequenceDim, FeatureDim]
type HeadsHiddenStatesTypedTensor[DType: Tensor, S: _SequenceDim] = TypedTensor[DType, BatchDim, HeadDim, S, HeadFeatureDim]
type HeadsAttentionTypedTensor[DType: Tensor, S0: _SequenceDim, S1: _SequenceDim] = TypedTensor[DType, BatchDim, HeadDim, S0, S1]
IdsTypedTensor = TypedTensor[torch.LongTensor, BatchDim, SequenceDim]
# fmt: on


@dataclass
class KeyValuePair[DType: Tensor, S: _SequenceDim]:
    key: HeadsHiddenStatesTypedTensor[DType, S]
    value: HeadsHiddenStatesTypedTensor[DType, S]


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    pass


@dataclass
class BaseModelOutputWithPastAndCrossAttentions[DType: Tensor](ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: Optional[HiddenStatesTypedTensor[DType]] = None
    past_key_values: Optional[List[KeyValuePair[DType, PastSequenceDim]]] = None
    hidden_states: Optional[Tuple[HiddenStatesTypedTensor[DType], ...]] = None
    attentions: Optional[Tuple[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim], ...]] = None
    cross_attentions: Optional[Tuple[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim], ...]] = None


@dataclass
class CausalLMOutputWithCrossAttentions[DType: Tensor](ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[TypedTensor[DType, BatchDim, SequenceDim, VocabDim]] = None
    past_key_values: Optional[List[KeyValuePair[DType, PastSequenceDim]]] = None
    hidden_states: Optional[Tuple[HiddenStatesTypedTensor[DType], ...]] = None
    attentions: Optional[Tuple[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim], ...]] = None
    cross_attentions: Optional[Tuple[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim], ...]] = None


class GPT2MlpGELUActivation[DType: Tensor](nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = lambda x: cast(DType, nn.functional.gelu(x))

    @staticmethod
    def _gelu_python(x: DType) -> DType:
        return cast(DType, x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))))
        # return cast(DType, 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))))

    def forward[*Ds](self, x: TypedTensor[DType, *Ds]) -> TypedTensor[DType, *Ds]:
        return x.transform(self.act)


@dataclasses.dataclass
class GPT2AttentionOutput[DType: Tensor]:
    attn_output: HiddenStatesTypedTensor[DType]
    present: Optional[KeyValuePair[DType, PastSequenceDim]]
    attn_weights: Optional[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim]]


class GPT2QueryKeyValueProjectionBase[DType: Tensor](nn.Module, abc.ABC):
    def __init__(self, embed_dim: int, split_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.split_size = split_size

    @abstractmethod
    def forward(
        self,
        hidden_states: HiddenStatesTypedTensor[DType],
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
    ) -> Tuple[HiddenStatesTypedTensor[DType], HiddenStatesTypedTensor[DType], HiddenStatesTypedTensor[DType]]:
        pass


class GPT2QueryKeyValueProjection[DType: Tensor](GPT2QueryKeyValueProjectionBase[DType]):
    def __init__(self, embed_dim: int, split_size: int):
        super().__init__(embed_dim, split_size)
        self.c_attn = tnn.Conv1D[DType, FeatureDim, TribbleFeatureDim](
            input_length=self.embed_dim, output_length=3 * self.embed_dim
        )

    def forward(
        self,
        hidden_states: HiddenStatesTypedTensor[DType],
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
    ) -> Tuple[HiddenStatesTypedTensor[DType], HiddenStatesTypedTensor[DType], HiddenStatesTypedTensor[DType]]:
        query, key, value = self.c_attn.forward(hidden_states).tensor.split(self.split_size, dim=2)
        return (
            TypedTensor[DType, BatchDim, SequenceDim, FeatureDim](query),
            TypedTensor[DType, BatchDim, SequenceDim, FeatureDim](key),
            TypedTensor[DType, BatchDim, SequenceDim, FeatureDim](value),
        )


class GPT2QueryKeyValueCrossAttentionProjection[DType: Tensor](GPT2QueryKeyValueProjectionBase[DType]):
    def __init__(self, embed_dim: int, split_size: int):
        super().__init__(embed_dim, split_size)
        self.c_attn = tnn.Conv1D[DType, FeatureDim, DoubleFeatureDim](
            input_length=self.embed_dim, output_length=2 * self.embed_dim
        )
        self.q_attn = tnn.Conv1D[DType, FeatureDim, FeatureDim](
            input_length=self.embed_dim, output_length=self.embed_dim
        )

    def forward(
        self,
        hidden_states: HiddenStatesTypedTensor[DType],
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
    ) -> Tuple[HiddenStatesTypedTensor[DType], HiddenStatesTypedTensor[DType], HiddenStatesTypedTensor[DType]]:
        if encoder_hidden_states is not None:
            query = self.q_attn.forward(hidden_states)
            key, value = self.c_attn.forward(encoder_hidden_states).tensor.split(self.split_size, dim=2)
            return TypedTensor(query.tensor), TypedTensor(key), TypedTensor(value)
        else:
            raise ValueError("encoder_hidden_states must not be None")


class GPT2Attention[DType: Tensor](nn.Module):
    def __init__(self, config: GPT2Config, is_cross_attention: bool = False, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        c_attn: GPT2QueryKeyValueProjectionBase[DType]
        if self.is_cross_attention:
            c_attn = GPT2QueryKeyValueCrossAttentionProjection[DType](
                embed_dim=self.embed_dim, split_size=self.split_size
            )
        else:
            c_attn = GPT2QueryKeyValueProjection[DType](embed_dim=self.embed_dim, split_size=self.split_size)
        self.c_attn = c_attn
        self.c_proj = tnn.Conv1D[DType, FeatureDim, FeatureDim](
            input_length=self.embed_dim, output_length=self.embed_dim
        )

        self.is_causal = True

    def _attn[PastAndCurrentSequenceDim: _SequenceDim](
        self,
        query: HeadsHiddenStatesTypedTensor[DType, SequenceDim],
        key: HeadsHiddenStatesTypedTensor[DType, PastAndCurrentSequenceDim],
        value: HeadsHiddenStatesTypedTensor[DType, PastAndCurrentSequenceDim],
        attention_mask: Optional[DType] = None,
        head_mask: Optional[DType] = None,
    ) -> Tuple[
        HeadsHiddenStatesTypedTensor[DType, SequenceDim],
        HeadsAttentionTypedTensor[DType, SequenceDim, PastAndCurrentSequenceDim],
    ]:
        # capturing runtime type
        _PastAndCurrentSequenceDim = key.args[1:][key.dim[PastAndCurrentSequenceDim]()]

        attn_weights = query.matmul(key.transpose[PastAndCurrentSequenceDim, HeadFeatureDim]())

        if self.scale_attn_weights:
            # value.size(-1) = head_features
            attn_weights = attn_weights / cast(
                DType, torch.full([], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device)
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            layer_idx = self.layer_idx
            attn_weights = attn_weights / float(layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            """
                                  key_length
                         _____________________________
                        |                             |
                        v                             v
                 .--->  1  0  0  0  0  0  0  0  0  0  0  .  .  .  .  .  .  .  .  .  .  .  .  .  .
   key_length    |      1  1  0  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 |      1  1  1  0  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 |      1  1  1  1  0  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 |      1  .  .  .  1  0  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 |      1  .  .  .  .  1  0  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 |     +-------------------------------+
                 | .-> |1  1  1  1  1  1  1  0  0  0  0| .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 | |   |1  1  1  1  1  1  1  1  0  0  0| .  .  .  .  .  .  .  .  .  .  .  .  .  .
   query_length  | |   |1  1  1  1  1  1  1  1  1  0  0| .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 | |   |1  1  1  1  1  1  1  1  1  1  0| .  .  .  .  .  .  .  .  .  .  .  .  .  .
                 `-`-> |1  1  1  1  1  1  1  1  1  1  1| .  .  .  .  .  .  .  .  .  .  .  .  .  .
                       +-------------------------------+
                        .  .  .  .  .  .  .  .  .  .  .  1  0  .  .  .  .  .  .  .  .  .  .  .  .
                        .  .  .  .  .  .  .  .  .  .  .  .  1  0  .  .  .  .  .  .  .  .  .  .  .
            """
            causal_mask = TypedTensor[
                torch.BoolTensor, Sub[BatchDim], Sub[HeadDim], Sub[SequenceDim], Sub[_PastAndCurrentSequenceDim]
            ](self.bias[:, :, key_length - query_length : key_length, :key_length])
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = TypedTensor[DType, Sub[_PastAndCurrentSequenceDim]](
                cast(DType, torch.full([1], mask_value, dtype=attn_weights.dtype, device=attn_weights.device))
            )
            attn_weights = ttorch.where(causal_mask, attn_weights, mask_value).shaped[
                BatchDim, HeadDim, SequenceDim, PastAndCurrentSequenceDim
            ]()

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.transform(lambda t: cast(DType, nn.functional.softmax(t, dim=-1)))

        # # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        # attn_weights = attn_weights.type(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # (batch, head, seq_length, past_seq_length + seq_length)
        # x (batch, head, past_seq_length + seq_length, head_features)
        # = (batch, head, seq_length, head_features)
        attn_output = attn_weights.matmul(value)

        return attn_output, attn_weights

    # def _upcast_and_reordered_attn(
    #         self,
    #         query: torch.Tensor,
    #         key: torch.Tensor,
    #         value: torch.Tensor,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
    #     bsz, num_heads, q_seq_len, dk = query.size()
    #     _, _, k_seq_len, _ = key.size()
    #
    #     # Preallocate attn_weights for `baddbmm`
    #     attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
    #
    #     # Compute Scale Factor
    #     scale_factor = 1.0
    #     if self.scale_attn_weights:
    #         scale_factor /= float(value.size(-1)) ** 0.5
    #
    #     if self.scale_attn_by_inverse_layer_idx:
    #         scale_factor /= float(self.layer_idx + 1)
    #
    #     # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
    #     with torch.amp.autocast(query.device.type, enabled=False):
    #         q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
    #         attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
    #         attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
    #
    #     if not self.is_cross_attention:
    #         # if only "normal" attention layer implements causal mask
    #         query_length, key_length = query.size(-2), key.size(-2)
    #         causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
    #         mask_value = torch.finfo(attn_weights.dtype).min
    #         # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    #         # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    #         mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    #         attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    #
    #     if attention_mask is not None:
    #         # Apply the attention mask
    #         attn_weights = attn_weights + attention_mask
    #
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    #
    #     # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
    #     if attn_weights.dtype != torch.float32:
    #         raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
    #     attn_weights = attn_weights.type(value.dtype)
    #
    #     # Mask heads if we want to
    #     if head_mask is not None:
    #         attn_weights = attn_weights * head_mask
    #
    #     attn_output = torch.matmul(attn_weights, value)
    #
    #     return attn_output, attn_weights

    def _split_heads(
        self, tensor: HiddenStatesTypedTensor[DType], num_heads: int, attn_head_size: int
    ) -> HeadsHiddenStatesTypedTensor[DType, SequenceDim]:
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        x = tensor.view[BatchDim, SequenceDim, HeadDim, HeadFeatureDim](Size(new_shape))
        return x.permute[BatchDim, HeadDim, SequenceDim, HeadFeatureDim]()

    def _merge_heads(
        self, tensor: HeadsHiddenStatesTypedTensor[DType, SequenceDim], num_heads: int, attn_head_size: int
    ) -> HiddenStatesTypedTensor[DType]:
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = tensor.permute[BatchDim, SequenceDim, HeadDim, HeadFeatureDim]().contiguous()
        new_shape = x.size()[:-2] + (num_heads * attn_head_size,)
        return x.view[BatchDim, SequenceDim, FeatureDim](Size(new_shape))

    def forward(
        self,
        hidden_states: HiddenStatesTypedTensor[DType],
        layer_past: Optional[KeyValuePair[DType, PastSequenceDim]] = None,
        attention_mask: Optional[DType] = None,
        head_mask: Optional[DType] = None,
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
        encoder_attention_mask: Optional[DType] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> GPT2AttentionOutput[DType]:
        if encoder_hidden_states is not None:
            if not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            attention_mask = encoder_attention_mask
        query, key, value = self.c_attn.forward(hidden_states, encoder_hidden_states)

        # (batch, head, seq_length, head_features)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        past_and_current_key: HeadsHiddenStatesTypedTensor[DType, PastSequenceDim]
        past_and_current_value: HeadsHiddenStatesTypedTensor[DType, PastSequenceDim]
        if layer_past is not None:
            past_and_current_key = ttorch.cat[_SequenceDim](
                [layer_past.key, key]
            ).shaped[BatchDim, HeadDim, PastSequenceDim, HeadFeatureDim]()
            past_and_current_value = ttorch.cat[_SequenceDim](
                [layer_past.value, value]
            ).shaped[BatchDim, HeadDim, PastSequenceDim, HeadFeatureDim]()
        else:
            past_and_current_key = key.shaped[BatchDim, HeadDim, PastSequenceDim, HeadFeatureDim]()
            past_and_current_value = value.shaped[BatchDim, HeadDim, PastSequenceDim, HeadFeatureDim]()

        # if self.reorder_and_upcast_attn:
        #     attn_output, attn_weights = self._upcast_and_reordered_attn(query, past_and_current_key, past_and_current_value, attention_mask, head_mask)
        # else:
        attn_output, attn_weights = self._attn(
            query, past_and_current_key, past_and_current_value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj.forward(attn_output)

        return GPT2AttentionOutput(
            attn_output,
            present=KeyValuePair[DType, PastSequenceDim](past_and_current_key, past_and_current_value)
            if use_cache
            else None,
            attn_weights=attn_weights if output_attentions else None,
        )


class GPT2MLP[DType: Tensor, Inner, D](nn.Module):
    def __init__(self, intermediate_size: int, embed_dim: int):
        super().__init__()
        self.c_fc = tnn.Conv1D[DType, D, Inner](input_length=embed_dim, output_length=intermediate_size)
        self.c_proj = tnn.Conv1D[DType, Inner, D](input_length=intermediate_size, output_length=embed_dim)
        self.act = GPT2MlpGELUActivation[DType]()

    def forward[*Ds](self, hidden_states: TypedTensor[DType, *Ds, D]) -> TypedTensor[DType, *Ds, D]:
        dtype, inner, d = self.__orig_class__.__args__
        hidden_states_0 = self.c_fc.forward(hidden_states)
        hidden_states_1 = self.act.forward(hidden_states_0)
        hidden_states_2 = self.c_proj.forward(hidden_states_1)
        hidden_states_2.args = hidden_states_2.args[:-1] + (d,)
        return hidden_states_2


@dataclasses.dataclass
class GPT2BlockOutput[DType: Tensor]:
    hidden_states: HiddenStatesTypedTensor[DType]
    present: Optional[KeyValuePair[DType, PastSequenceDim]]
    attentions: Optional[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim]]
    cross_attentions: Optional[HeadsAttentionTypedTensor[DType, SequenceDim, PastSequenceDim]]


class GPT2Block[DType: Tensor](nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: Optional[int] = None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = tnn.LayerNorm[DType, FeatureDim](hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention[DType](config=config, layer_idx=layer_idx)
        self.ln_2 = tnn.LayerNorm[DType, FeatureDim](hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention[DType](config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = tnn.LayerNorm[DType, FeatureDim](hidden_size, eps=config.layer_norm_epsilon)

        class InnerDim(Dimension, length=inner_dim):
            pass  # noqa E701

        self.mlp = GPT2MLP[DType, InnerDim, FeatureDim](inner_dim, config.hidden_size)

    def forward(
        self,
        hidden_states: HiddenStatesTypedTensor[DType],
        layer_past: Optional[KeyValuePair[DType, PastSequenceDim]] = None,
        attention_mask: Optional[DType] = None,
        head_mask: Optional[DType] = None,
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
        encoder_attention_mask: Optional[DType] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> GPT2BlockOutput[DType]:
        # Attention
        @tnn.residual_connection
        def attention(hidden_states: HiddenStatesTypedTensor[DType]):
            attn_outputs = self.attn.forward(
                self.ln_1.forward(hidden_states),
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            return attn_outputs.attn_output, attn_outputs

        hidden_states_1, attn_outputs = attention(hidden_states)

        # Cross-Attention
        cross_attn_outputs: Optional[GPT2AttentionOutput[DType]] = None
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )

            @tnn.residual_connection
            def crossattention(hidden_states: HiddenStatesTypedTensor[DType]):
                cross_attn_outputs = self.crossattention.forward(
                    self.ln_cross_attn.forward(hidden_states),
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                return cross_attn_outputs.attn_output, cross_attn_outputs

            hidden_states_2, cross_attn_outputs = crossattention(hidden_states_1)
        else:
            hidden_states_2 = hidden_states_1

        # Feed Forward
        @tnn.residual_connection
        def feedforward(hidden_states: HiddenStatesTypedTensor[DType]):
            feed_forward_hidden_states = self.mlp.forward(self.ln_2.forward(hidden_states))
            return feed_forward_hidden_states, feed_forward_hidden_states

        hidden_states_3, _ = feedforward(hidden_states_2)

        return GPT2BlockOutput(
            hidden_states_3,
            attn_outputs.present if use_cache else None,
            attn_outputs.attn_weights,
            # add cross attentions if we output attention weights
            cross_attn_outputs.attn_weights if cross_attn_outputs is not None else None,
        )  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel[DType: Tensor](PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, tnn.Conv1D):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight_t.tensor = module.weight
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, tnn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.linear.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.linear.bias is not None:
                module.linear.bias.data.zero_()
        # elif isinstance(module, nn.Linear):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        elif isinstance(module, tnn.Embedding):
            module.embedding.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.embedding.padding_idx is not None:
                module.embedding.weight.data[module.embedding.padding_idx].zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, tnn.LayerNorm):
            module.ln.bias.data.zero_()
            module.ln.weight.data.fill_(1.0)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


class GPT2Model[DType: Tensor](GPT2PreTrainedModel[DType]):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = tnn.Embedding[DType, SequenceDim, FeatureDim](config.vocab_size, self.embed_dim)
        self.wpe = tnn.Embedding[DType, SequenceDim, FeatureDim](config.max_position_embeddings, self.embed_dim)

        self.hs = [GPT2Block[DType](config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.h = nn.ModuleList(self.hs)
        self.ln_f = tnn.LayerNorm[DType, FeatureDim](self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte.embedding

    def set_input_embeddings(self, new_embeddings):
        self.wte.embedding = new_embeddings

    def forward(
        self,
        input_ids: Optional[IdsTypedTensor] = None,
        past_key_values: Optional[List[Optional[KeyValuePair[DType, PastSequenceDim]]]] = None,
        attention_mask: Optional[DType] = None,
        token_type_ids: Optional[IdsTypedTensor] = None,
        position_ids: Optional[IdsTypedTensor] = None,
        head_mask: Optional[DType] = None,
        inputs_embeds: Optional[TypedTensor[DType, BatchDim, SequenceDim, FeatureDim]] = None,
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
        encoder_attention_mask: Optional[DType] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions[DType]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            batch_size = input_shape[0]
            input_seq_length = input_shape[1]
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = input_shape[0]
            input_seq_length = input_shape[1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        _past_key_values: List[Optional[KeyValuePair[DType, PastSequenceDim]]]
        past_seq_length: int = 0
        if past_key_values is None:
            past_seq_length = 0
            _past_key_values = [None] * len(self.h)
        else:
            if past_key_values[0] is not None:
                past_seq_length = past_key_values[0].key.size(-2)
            _past_key_values = past_key_values

        if position_ids is None:
            position_ids_tensor = torch.arange(
                past_seq_length, input_seq_length + past_seq_length, dtype=torch.long, device=device
            )
            position_ids = TypedTensor[torch.LongTensor, BatchDim, SequenceDim](
                cast(torch.LongTensor, position_ids_tensor.unsqueeze(0).expand(batch_size, -1))
            )

        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.wte.forward(input_ids)
            else:
                raise ValueError()
        position_embeds = self.wpe.forward(position_ids)
        hidden_states = inputs_embeds + cast(DType, position_embeds.tensor)

        # Attention mask.
        if attention_mask is not None:
            _attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            _attention_mask = _attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            _attention_mask = _attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            _attention_mask = (1.0 - _attention_mask) * torch.finfo(self.dtype).min
            attention_mask = cast(DType, _attention_mask)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = cast(DType, torch.ones(encoder_hidden_shape, device=device))
            encoder_attention_mask = cast(DType, self.invert_attention_mask(encoder_attention_mask))
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        _head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte.forward(token_type_ids)
            hidden_states = hidden_states + token_type_embeds.tensor

        # output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.hs, _past_key_values)):
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block.forward(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=cast(DType, _head_mask[i]) if _head_mask[i] is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs.hidden_states
            if presents is not None:
                presents.append(outputs.present)

            if output_attentions:
                if all_self_attentions is not None and outputs.attentions is not None:
                    all_self_attentions = all_self_attentions + (outputs.attentions,)
                if (
                    self.config.add_cross_attention
                    and all_cross_attentions is not None
                    and outputs.cross_attentions is not None
                ):
                    all_cross_attentions = all_cross_attentions + (outputs.cross_attentions,)

        hidden_states = self.ln_f.forward(hidden_states)

        # hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2LMHeadModel[DType: Tensor](GPT2PreTrainedModel[DType]):
    _tied_weights_keys = ["lm_head.linear.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model[DType](config)
        self.lm_head = tnn.Linear[DType, FeatureDim, VocabDim](config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.linear = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: IdsTypedTensor,
        past_key_values: Optional[List[KeyValuePair[DType, PastSequenceDim]]] = None,
        inputs_embeds: Optional[HiddenStatesTypedTensor[DType]] = None,
        **kwargs,
    ):
        token_type_ids: Optional[IdsTypedTensor] = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0].key.size()[2]  # (batch, head, seq_length, head_features)

            # Some generation methods already pass only the last input ID
            if input_ids.size()[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.size()[1] - 1

            input_ids = input_ids.transform(lambda t: cast(torch.LongTensor, t[:, remove_prefix_length:]))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.transform(
                    lambda t: cast(torch.LongTensor, t[:, -input_ids.size()[1] :])
                )

        attention_mask: Optional[DType] = kwargs.get("attention_mask", None)
        position_ids: Optional[IdsTypedTensor] = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            _position_ids = attention_mask.long().cumsum(-1) - 1
            _position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                _position_ids = _position_ids[:, -input_ids.size()[1] :]
            position_ids = IdsTypedTensor(cast(torch.LongTensor, _position_ids))
        else:
            position_ids = None

        model_inputs: dict[str, Any] = {}
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: Optional[IdsTypedTensor] = None,
        past_key_values: Optional[List[Optional[KeyValuePair[DType, PastSequenceDim]]]] = None,
        attention_mask: Optional[DType] = None,
        token_type_ids: Optional[IdsTypedTensor] = None,
        position_ids: Optional[IdsTypedTensor] = None,
        head_mask: Optional[DType] = None,
        inputs_embeds: Optional[HiddenStatesTypedTensor[DType]] = None,
        encoder_hidden_states: Optional[HiddenStatesTypedTensor[DType]] = None,
        encoder_attention_mask: Optional[DType] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions[DType]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        transformer_outputs = self.transformer.forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        if hidden_states is None:
            raise ValueError()

        lm_logits = self.lm_head.forward(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


if __name__ == "__main__":
    import random

    import numpy as np
    import torch
    import transformers

    config = transformers.GPT2Config(
        vocab_size=6,
        n_positions=5,
        n_embd=30,
        n_layer=5,
        n_head=3,
        n_inner=None,
        activation_function="gelu",
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=5,
        eos_token_id=5,
        attn_implementation="eager",
        return_dict=True,
        output_hidden_states=True,
        output_attentions=True,
        add_cross_attention=False,
        pruned_heads=None,
    )

    input_ids = cast(torch.LongTensor, torch.LongTensor([[1, 2, 3], [4, 4, 2]]))
    batch_features_t = torch.randn((2, 30), dtype=torch.float)
    batch_seq_features_t = torch.randn((2, 3, 30), dtype=torch.float)

    def reset_state():
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    ###########################################################################
    reset_state()

    subject_model = GPT2LMHeadModel[torch.FloatTensor](config).eval()
    subject_res = subject_model.forward(
        input_ids=TypedTensor[torch.LongTensor, BatchDim, SequenceDim](input_ids),
        use_cache=True,
    )
    ###########################################################################
    reset_state()

    baseline_model = transformers.GPT2LMHeadModel(config).eval()
    baseline_res = baseline_model.forward(
        input_ids=input_ids,
        use_cache=True,
    )
    ###########################################################################

    assert isinstance(baseline_res, transformers.modeling_outputs.CausalLMOutputWithCrossAttentions)
    assert subject_res.attentions is not None and baseline_res.attentions is not None
    assert torch.equal(subject_res.attentions[0].tensor, baseline_res.attentions[0])

    assert subject_res.hidden_states is not None and baseline_res.hidden_states is not None
    assert torch.equal(subject_res.hidden_states[0].tensor, baseline_res.hidden_states[0])

    assert subject_res.logits is not None and baseline_res.logits is not None
    assert torch.equal(subject_res.logits.tensor, baseline_res.logits)

    assert subject_res.past_key_values is not None and baseline_res.past_key_values is not None
    assert torch.equal(subject_res.past_key_values[0].key.tensor, baseline_res.past_key_values[0][0])
    assert torch.equal(subject_res.past_key_values[0].value.tensor, baseline_res.past_key_values[0][1])  # type: ignore

    print("OK")
