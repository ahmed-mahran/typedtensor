from typing import Literal, Type, cast, overload

from mypyright_extensions import Map, subscriptablefunction
import torch
from torch import BoolTensor, Tensor

from ..dimension import Concat, Dimension, Rec, Sub
from ..shape_info import Broadcast, Shape, ShapeInfo
from ..typed_tensor import TypedTensor


def addmm[DType: Tensor, *Ds, D0, D1, D2](
    input: torch.Tensor, mat1: TypedTensor[DType, *Ds, D0, D1], mat2: TypedTensor[DType, *Ds, D1, D2]
) -> TypedTensor[DType, *Ds, D0, D2]:
    ts1 = list(mat1.args)
    ts2 = list(mat2.args)
    ts = ts1[:-2] + [ts1[-2], ts2[-1]]
    res = cast(DType, torch.addmm(input, mat1.tensor, mat2.tensor))
    return TypedTensor(res, tuple(ts))


@subscriptablefunction
def cat[DType: Tensor, *Init, D: Dimension, *Tail](
    tp: Type[D],
    xs: list[TypedTensor[DType, *Init, D, *Tail]]
) -> TypedTensor[DType, *Init, Rec[D, Concat], *Tail]:
    """
    cat[Seq](xs)
    """
    dim = xs[0].dim[tp]()
    args = list(xs[0].args)
    args[dim + 1] = Rec[tp, Concat]
    return TypedTensor[DType, *Init, Rec[D, Concat], *Tail](
        cast(DType, torch.cat([x.tensor for x in xs], dim=dim)), tuple(args)
    )

# def stack[DType: Tensor, D: Dimension, I: IntLiteral, *Ds](
#     tp: Type[D],:
#     xs: list[TypedTensor[DType, *Ds[:I], *Ds[I:]]],
# ) -> TypedTensor[DType, *Ds[:I], D, *Ds[I:]]:
@overload
@subscriptablefunction
def stack[DType: Tensor, D: Dimension, *Tail](
    tp: Type[D],
    xs: list[TypedTensor[DType, *Tail]],
) -> TypedTensor[DType, D, *Tail]: ...

@overload
@subscriptablefunction
def stack[DType: Tensor, After, D: Dimension, *Init, *Tail](
    tp: Map[Type, After, D],
    xs: list[TypedTensor[DType, *Init, After, *Tail]],
) -> TypedTensor[DType, *Init, After, D, *Tail]: ...

@subscriptablefunction
def stack[DType: Tensor, After, D: Dimension, *Init, *Tail](
    tp: Map[Type, After, D] | Type[D],
    xs: list[TypedTensor[DType, *Init, After, *Tail]] | list[TypedTensor[DType, *Tail]],
) -> TypedTensor[DType, *Init, After, D, *Tail] | TypedTensor[DType, D, *Tail]:
    """
    stack[Batch](xs)
    """
    if isinstance(tp, tuple):
        after, d = tp
        dim = xs[0].dim[after]()
    else:
        dim = 0
    args = [i for i in xs[0].args]
    index = dim + 1
    args = args[:index] + [d] + args[index:]
    return TypedTensor(
        cast(DType, torch.stack([x.tensor for x in xs], dim=dim)), tuple(args)
    )

@overload
@subscriptablefunction
def column_stack[DType: Tensor, D0, D1: Dimension](
    tp: Type[D1], xs: tuple[TypedTensor[DType, D0] | TypedTensor[DType, D0, D1], ...],
) -> TypedTensor[DType, D0, Rec[D1, Concat]]: ...

@overload
@subscriptablefunction
def column_stack[DType: Tensor, D0, D1: Dimension, *Ds](
    tp: Type[D1], xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...],
) -> TypedTensor[DType, D0, Rec[D1, Concat], *Ds]: ...

@subscriptablefunction
def column_stack[DType: Tensor, D0, D1: Dimension, *Ds](
    tp: Type[D1],
    xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...] | tuple[TypedTensor[DType, D0] | TypedTensor[DType, D0, D1], ...],
) -> TypedTensor[DType, D0, Rec[D1, Concat], *Ds] | TypedTensor[DType, D0, Rec[D1, Concat]]:
    """
    column_stack[Sequence](xs)
    """
    args = list(max(map(lambda x: x.args, xs), key=len))
    dim = 1
    index = dim + 1
    args = args[:index] + [Rec[tp, Concat]] + args[index + 1 :]
    return TypedTensor(cast(DType, torch.column_stack([x.tensor for x in xs])), tuple(args))

@overload
@subscriptablefunction
def dstack[DType: Tensor, D0, D2: Dimension](
    tp: Type[D2],
    xs: tuple[TypedTensor[DType, D0], ...],
) -> TypedTensor[DType, Dimension, D0, Rec[D2, Concat]]: ...

@overload
@subscriptablefunction
def dstack[DType: Tensor, D0, D1, D2: Dimension](
    tp: Type[D2],
    xs: tuple[TypedTensor[DType, D0, D1] | TypedTensor[DType, D0, D1, D2], ...],
) -> TypedTensor[DType, D0, D1, Rec[D2, Concat]]: ...

@overload
@subscriptablefunction
def dstack[DType: Tensor, D0, D1, D2: Dimension, *Ds](
    tp: Type[D2],
    xs: tuple[TypedTensor[DType, D0, D1, D2, *Ds], ...],
) -> TypedTensor[DType, D0, D1, Rec[D2, Concat], *Ds]: ...

@subscriptablefunction
def dstack[DType: Tensor, D0, D1, D2: Dimension, *Ds](
    tp: Type[D2],
    xs: tuple[TypedTensor[DType, D0, D1, D2, *Ds], ...]
    | tuple[TypedTensor[DType, D0, D1] | TypedTensor[DType, D0, D1, D2], ...]
    | tuple[TypedTensor[DType, D0], ...],
) -> (
    TypedTensor[DType, D0, D1, Rec[D2, Concat], *Ds]
    | TypedTensor[DType, D0, D1, Rec[D2, Concat]]
    | TypedTensor[DType, Dimension, D0, Rec[D2, Concat]]
):
    """
    dstack[Feature](xs)
    """
    args = list(max(map(lambda x: x.args, xs), key=len))
    if len(args) > 2:
        dim = 2
        index = dim + 1
        args = args[:index] + [Rec[tp, Concat]] + args[index + 1 :]
    else:
        args = [args[0], Dimension, args[1], Rec[tp, Concat]]
    return TypedTensor(cast(DType, torch.dstack([x.tensor for x in xs])), tuple(args))


@overload
@subscriptablefunction
def hstack[DType: Tensor, D: Dimension](
    tp: Type[D],
    xs: tuple[TypedTensor[DType, D], ...],
) -> TypedTensor[DType, Rec[D, Concat]]: ...

@overload
@subscriptablefunction
def hstack[DType: Tensor, D0, D: Dimension, *Ds](
    tp: Type[D],
    xs: tuple[TypedTensor[DType, D0, D, *Ds], ...],
) -> TypedTensor[DType, D0, Rec[D, Concat], *Ds]: ...

@subscriptablefunction
def hstack[DType: Tensor, D0, D: Dimension, *Ds](
    tp: Type[D],
    xs: tuple[TypedTensor[DType, D0, D, *Ds], ...] | tuple[TypedTensor[DType, D], ...],
) -> TypedTensor[DType, D0, Rec[D, Concat], *Ds] | TypedTensor[DType, Rec[D, Concat]]:
    """
    hstack[D](xs)
    """
    args = list(xs[0].args[:])
    dim = 1 if len(args) > 2 else 0
    index = dim + 1
    args[index] = Rec[tp, Concat]
    return TypedTensor(cast(DType, torch.hstack([x.tensor for x in xs])), tuple(args))

@overload
@subscriptablefunction
def row_stack[DType: Tensor, D0: Dimension, D1](
    tp: Type[D0],
    xs: tuple[TypedTensor[DType, D1], ...],
) -> TypedTensor[DType, Rec[D0, Concat], D1]: ...

@overload
@subscriptablefunction
def row_stack[DType: Tensor, D0: Dimension, D1](
    tp: Type[D0],
    xs: tuple[TypedTensor[DType, D1] | TypedTensor[DType, D0, D1], ...],
) -> TypedTensor[DType, Rec[D0, Concat], D1]: ...

@overload
@subscriptablefunction
def row_stack[DType: Tensor, D0: Dimension, D1, *Ds](
    tp: Type[D0],
    xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...],
) -> TypedTensor[DType, Rec[D0, Concat], D1, *Ds]: ...

@subscriptablefunction
def row_stack[DType: Tensor, D0: Dimension, D1, *Ds](
    tp: Type[D0],
    xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...]
    | tuple[TypedTensor[DType, D1] | TypedTensor[DType, D0, D1], ...]
    | tuple[TypedTensor[DType, D1], ...],
) -> (
    TypedTensor[DType, Rec[D0, Concat], D1, *Ds]
    | TypedTensor[DType, Rec[D0, Concat], D1]
    | TypedTensor[DType, Rec[D0, Concat], D1]
):
    """
    row_stack[Batch](xs)
    """
    args = list(max(map(lambda x: x.args, xs), key=len))
    if len(args) > 2:
        dim = 0
        index = dim + 1
        args[index] = Rec[tp, Concat]
    else:
        args = [args[0], Rec[tp, Concat], args[1]]
    return TypedTensor(cast(DType, torch.vstack([x.tensor for x in xs])), tuple(args))

"""
vstack[Batch](xs)
"""
vstack = row_stack


@overload
@subscriptablefunction
def sum[DType: Tensor, *Init, D: Dimension, *Tail](
    tp: Type[D],
    x: TypedTensor[DType, *Init, D, *Tail], keepdim: Literal[True]
) -> TypedTensor[DType, *Init, Sub[D], *Tail]: ...

@overload
@subscriptablefunction
def sum[DType: Tensor, *Init, D: Dimension, *Tail](
    tp: Type[D],
    x: TypedTensor[DType, *Init, D, *Tail], keepdim: Literal[False]
) -> TypedTensor[DType, *Init, *Tail]: ...

@subscriptablefunction
def sum[DType: Tensor, *Init, D: Dimension, *Tail](
    tp: Type[D],
    x: TypedTensor[DType, *Init, D, *Tail], keepdim: Literal[False] | Literal[True]
) -> TypedTensor[DType, *Init, *Tail] | TypedTensor[DType, *Init, Sub[D], *Tail]:
    """
    sum[D](xs)
    """
    args = list(x.args)
    dim = x.dim[tp]()
    index = dim + 1
    args = args[:index] + ([Sub[tp]] if keepdim else []) + args[index + 1 :]
    return TypedTensor(cast(DType, torch.sum(x.tensor, dim=dim, keepdim=keepdim)), tuple(args))

@subscriptablefunction
def transpose[DType: Tensor, *Init, D0, *Mid, D1, *Tail](
    shape: Map[Type, D0, D1],
    x: TypedTensor[DType, *Init, D0, *Mid, D1, *Tail]
) -> TypedTensor[DType, *Init, D1, *Mid, D0, *Tail]:
    """
    transpose[D0, D1](xs)
    """
    d0, d1 = shape
    dim0, dim1 = x.dim[d0](), x.dim[d1]()
    ts = list(x.args[1:])
    d0, d1 = ts[dim0], ts[dim1]  # we prefer concrete types from tensor definition
    ts[dim0] = d1
    ts[dim1] = d0
    return TypedTensor(cast(DType, x.tensor.transpose(dim0, dim1)), tuple([x.args[0]] + ts))

def where[DType: Tensor, *Cs, *Is, *Os](
    condition: TypedTensor[BoolTensor, *Cs], input: TypedTensor[DType, *Is], other: TypedTensor[DType, *Os]
) -> TypedTensor[DType, Broadcast[Shape[Broadcast[Shape[*Cs], Shape[*Is]]], Shape[*Os]]]:
    res = torch.where(condition.tensor, input.tensor, other.tensor)
    broadcast_shape = ShapeInfo(
        Broadcast.broadcast(Broadcast.broadcast(condition.shape.args, input.shape.args), other.shape.args)
    )
    return TypedTensor(cast(DType, res), (input.args[0],) + tuple(broadcast_shape.types()))
