from typing import cast

import torch
from torch import Tensor

from .typed_tensor import TypedTensor


def addmm[DType: Tensor, *Ds, D0, D1, D2](
    input: torch.Tensor, mat1: TypedTensor[DType, *Ds, D0, D1], mat2: TypedTensor[DType, *Ds, D1, D2]
) -> TypedTensor[DType, *Ds, D0, D2]:
    ts1 = list(mat1.args)
    ts2 = list(mat2.args)
    ts = ts1[:-3] + [ts1[-2], ts2[-1]]
    res = cast(DType, torch.addmm(input, mat1.tensor, mat2.tensor))
    return TypedTensor(res, tuple(ts))
