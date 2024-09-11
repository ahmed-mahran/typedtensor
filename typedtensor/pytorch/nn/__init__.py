from typing import Callable, List, Optional, Tuple, Union, cast

import torch
from torch import Size, Tensor, nn

from ... import pytorch as ttorch
from ...dimension import Dimension
from ...typed_tensor import TypedTensor
from ...utils import CapturedTypeArgs

_shape_t = Union[int, List[int], Size]


class Linear[DType: Tensor, D0: Dimension, D1: Dimension](nn.Module, CapturedTypeArgs):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)

    def forward[*Ds](self, x: TypedTensor[DType, *Ds, D0]) -> TypedTensor[DType, *Ds, D1]:
        dtype, d0, d1 = self.type_args
        return TypedTensor(cast(DType, self.linear(x.tensor)), x.args[:-1] + (d1,))


class Conv1D[DType: Tensor, D0, D1](nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        input_length (`int`): The number of input features.
        output_length (`int`): The number of output features.
    """

    def __init__(self, input_length: int, output_length: int):
        super().__init__()
        self.output_length = output_length
        self.weight = nn.Parameter(torch.empty(input_length, output_length))
        self.bias = nn.Parameter(torch.zeros(output_length))
        nn.init.normal_(self.weight, std=0.02)
        self.weight_t = TypedTensor[DType, D0, D1](cast(DType, self.weight))

    def forward[*Ds](self, x: TypedTensor[DType, *Ds, D0]) -> TypedTensor[DType, *Ds, D1]:
        # dtype, d0, d1 = self.__orig_class__.__args__
        size_out = x.size()[:-1] + (self.output_length,)
        # shape_out = x.args[1:-1] + (d1,)
        x_as_2d = x.view[Dimension, D0](Size((-1, x.size(-1))))
        x_out_as_2d = ttorch.addmm(self.bias, x_as_2d, self.weight_t)
        x_out = x_out_as_2d.view[*Ds, D1](Size(size_out))
        return x_out
        # return x_out_as_2d.view[*Ds, D1](Size(size_out))
        # return x_out_as_2d.view(Shape[*Ds, D1], Size(size_out), shape_out)


class LayerNorm[DType: Tensor, D0](nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps, elementwise_affine, bias, device, dtype)

    def forward[*Ds](self, x: TypedTensor[DType, *Ds, D0]) -> TypedTensor[DType, *Ds, D0]:
        return x.transform(lambda t: cast(DType, self.ln.forward(t)))


class Embedding[DType: Tensor, IdsDim: Dimension, EmbeddingDim: Dimension](nn.Module, CapturedTypeArgs):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )

    def forward[*Ds](
        self, x: TypedTensor[torch.LongTensor, *Ds, IdsDim]
    ) -> TypedTensor[DType, *Ds, IdsDim, EmbeddingDim]:
        dtype, _, embd = self.type_args
        args = (dtype,) + x.args[1:] + (embd,)
        return TypedTensor(cast(DType, self.embedding.forward(x.tensor)), args)


def residual_connection[DType: Tensor, *Ds, R](
    block: Callable[[TypedTensor[DType, *Ds]], Tuple[TypedTensor[DType, *Ds], R]],
):
    def inner(x: TypedTensor[DType, *Ds]) -> Tuple[TypedTensor[DType, *Ds], R]:
        residual = x.tensor
        y, result = block(x)
        return y + residual, result

    return inner
