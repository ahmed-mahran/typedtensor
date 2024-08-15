from typing import Type, cast, overload

import torch
from torch import BoolTensor, Tensor

from ..dimension import Concat, Dimension, Rec, Z
from ..shape_info import Broadcast, Shape, ShapeInfo
from ..typed_tensor import TypedTensor


def addmm[DType: Tensor, *Ds, D0, D1, D2](
    input: torch.Tensor, mat1: TypedTensor[DType, *Ds, D0, D1], mat2: TypedTensor[DType, *Ds, D1, D2]
) -> TypedTensor[DType, *Ds, D0, D2]:
    ts1 = list(mat1.args)
    ts2 = list(mat2.args)
    ts = ts1[:-3] + [ts1[-2], ts2[-1]]
    res = cast(DType, torch.addmm(input, mat1.tensor, mat2.tensor))
    return TypedTensor(res, tuple(ts))


class _cat:
    def __getitem__[D: Dimension](self, tp: Type[D]):
        def inner[DType: Tensor](
            xs: list[TypedTensor[DType, Z[Dimension], D, Z[Dimension]]],
        ) -> TypedTensor[DType, Z[Dimension], Rec[D, Concat], Z[Dimension]]:
            dim = xs[0].dim[tp]
            args = list(xs[0].args)
            args[dim + 1] = Rec[tp, Concat]
            return TypedTensor[DType, Z[Dimension], Rec[D, Concat], Z[Dimension]](
                cast(DType, torch.cat([x.tensor for x in xs], dim=dim)), tuple(args)
            )

        return inner


"""
cat[Seq](xs)
"""
cat = _cat()


class _stack:
    def __getitem__[D: Dimension](self, tp: Type[D]):
        def inner[DType: Tensor, *Ds](
            xs: list[TypedTensor[DType, *Ds]],
            dim: int = 0,
        ) -> TypedTensor[DType, Z[Dimension], D, Z[Dimension]]:
            args = [i for i in xs[0].args]
            index = dim + 1
            args = args[:index] + [tp] + args[index:]
            return TypedTensor[DType, Z[Dimension], D, Z[Dimension]](
                cast(DType, torch.stack([x.tensor for x in xs], dim=dim)), tuple(args)
            )

        return inner


"""
stack[Batch](xs, dim=0)
"""
stack = _stack()


def where[DType: Tensor, *Cs, *Is, *Os](
    condition: TypedTensor[BoolTensor, *Cs], input: TypedTensor[DType, *Is], other: TypedTensor[DType, *Os]
):
    res = torch.where(condition.tensor, input.tensor, other.tensor)
    broadcast_shape = ShapeInfo(
        Broadcast.broadcast(Broadcast.broadcast(condition.shape.args, input.shape.args), other.shape.args)
    )
    return TypedTensor[DType, Broadcast[Shape[Broadcast[Shape[*Cs], Shape[*Is]]], Shape[*Os]]](
        cast(DType, res), (input.args[0],) + tuple(broadcast_shape.types())
    )
