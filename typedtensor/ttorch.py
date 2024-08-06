from typing import Type, cast

import torch
from torch import Tensor

from .dimension import Concat, Dimension, Rec, Z
from .typed_tensor import TypedTensor


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
