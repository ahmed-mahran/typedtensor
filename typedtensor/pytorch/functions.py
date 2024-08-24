from typing import Literal, Type, cast, overload

import torch
from torch import BoolTensor, Tensor

from ..dimension import Concat, Dimension, Rec, Z, Sub
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


class _column_stack:
    def __getitem__[D1: Dimension](self, tp: Type[D1]):
        @overload
        def inner[DType: Tensor, D0](
            xs: tuple[TypedTensor[DType, D0] | TypedTensor[DType, D0, D1], ...],
        ) -> TypedTensor[DType, D0, Rec[D1, Concat]]: ...

        @overload
        def inner[DType: Tensor, D0, *Ds](
            xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...],
        ) -> TypedTensor[DType, D0, Rec[D1, Concat], *Ds]: ...

        def inner[DType: Tensor, D0, *Ds](
            xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...]
            | tuple[TypedTensor[DType, D0] | TypedTensor[DType, D0, D1], ...],
        ) -> TypedTensor[DType, D0, Rec[D1, Concat], *Ds] | TypedTensor[DType, D0, Rec[D1, Concat]]:
            args = list(max(map(lambda x: x.args, xs), key=len))
            dim = 1
            index = dim + 1
            args = args[:index] + [Rec[tp, Concat]] + args[index + 1 :]
            return TypedTensor(cast(DType, torch.column_stack([x.tensor for x in xs])), tuple(args))

        return inner


"""
column_stack[Sequence](xs)
"""
column_stack = _column_stack()


class _dstack:
    def __getitem__[D2: Dimension](self, tp: Type[D2]):
        @overload
        def inner[DType: Tensor, D0](
            xs: tuple[TypedTensor[DType, D0], ...],
        ) -> TypedTensor[DType, Dimension, D0, Rec[D2, Concat]]: ...

        @overload
        def inner[DType: Tensor, D0, D1](
            xs: tuple[TypedTensor[DType, D0, D1] | TypedTensor[DType, D0, D1, D2], ...],
        ) -> TypedTensor[DType, D0, D1, Rec[D2, Concat]]: ...

        @overload
        def inner[DType: Tensor, D0, D1, *Ds](
            xs: tuple[TypedTensor[DType, D0, D1, D2, *Ds], ...],
        ) -> TypedTensor[DType, D0, D1, Rec[D2, Concat], *Ds]: ...

        def inner[DType: Tensor, D0, D1, *Ds](
            xs: tuple[TypedTensor[DType, D0, D1, D2, *Ds], ...]
            | tuple[TypedTensor[DType, D0, D1] | TypedTensor[DType, D0, D1, D2], ...]
            | tuple[TypedTensor[DType, D0], ...],
        ) -> (
            TypedTensor[DType, D0, D1, Rec[D2, Concat], *Ds]
            | TypedTensor[DType, D0, D1, Rec[D2, Concat]]
            | TypedTensor[DType, Dimension, D0, Rec[D2, Concat]]
        ):
            args = list(max(map(lambda x: x.args, xs), key=len))
            if len(args) > 2:
                dim = 2
                index = dim + 1
                args = args[:index] + [Rec[tp, Concat]] + args[index + 1 :]
            else:
                args = [args[0], Dimension, args[1], Rec[tp, Concat]]
            return TypedTensor(cast(DType, torch.dstack([x.tensor for x in xs])), tuple(args))

        return inner


"""
dstack[Feature](xs)
"""
dstack = _dstack()


class _hstack:
    def __getitem__[D: Dimension](self, tp: Type[D]):
        @overload
        def inner[DType: Tensor](
            xs: tuple[TypedTensor[DType, D], ...],
        ) -> TypedTensor[DType, Rec[D, Concat]]: ...

        @overload
        def inner[DType: Tensor, D0, *Ds](
            xs: tuple[TypedTensor[DType, D0, D, *Ds], ...],
        ) -> TypedTensor[DType, D0, Rec[D, Concat], *Ds]: ...

        def inner[DType: Tensor, D0, *Ds](
            xs: tuple[TypedTensor[DType, D0, D, *Ds], ...] | tuple[TypedTensor[DType, D], ...],
        ) -> TypedTensor[DType, D0, Rec[D, Concat], *Ds] | TypedTensor[DType, Rec[D, Concat]]:
            args = list(xs[0].args[:])
            dim = 1 if len(args) > 2 else 0
            index = dim + 1
            args[index] = Rec[tp, Concat]
            return TypedTensor(cast(DType, torch.hstack([x.tensor for x in xs])), tuple(args))

        return inner


"""
hstack[D](xs)
"""
hstack = _hstack()


class _vstack:
    def __getitem__[D0: Dimension](self, tp: Type[D0]):
        @overload
        def inner[DType: Tensor, D1](
            xs: tuple[TypedTensor[DType, D1], ...],
        ) -> TypedTensor[DType, Rec[D0, Concat], D1]: ...

        @overload
        def inner[DType: Tensor, D1](
            xs: tuple[TypedTensor[DType, D1] | TypedTensor[DType, D0, D1], ...],
        ) -> TypedTensor[DType, Rec[D0, Concat], D1]: ...

        @overload
        def inner[DType: Tensor, D1, *Ds](
            xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...],
        ) -> TypedTensor[DType, Rec[D0, Concat], D1, *Ds]: ...

        def inner[DType: Tensor, D1, *Ds](
            xs: tuple[TypedTensor[DType, D0, D1, *Ds], ...]
            | tuple[TypedTensor[DType, D1] | TypedTensor[DType, D0, D1], ...]
            | tuple[TypedTensor[DType, D1], ...],
        ) -> (
            TypedTensor[DType, Rec[D0, Concat], D1, *Ds]
            | TypedTensor[DType, Rec[D0, Concat], D1]
            | TypedTensor[DType, Rec[D0, Concat], D1]
        ):
            args = list(max(map(lambda x: x.args, xs), key=len))
            if len(args) > 2:
                dim = 0
                index = dim + 1
                args[index] = Rec[tp, Concat]
            else:
                args = [args[0], Rec[tp, Concat], args[1]]
            return TypedTensor(cast(DType, torch.vstack([x.tensor for x in xs])), tuple(args))

        return inner


"""
row_stack[Batch](xs)
"""
row_stack = _vstack()

"""
vstack[Batch](xs)
"""
vstack = _vstack()


class _sum:
    def __getitem__[D: Dimension](self, tp: Type[D]):
        @overload
        def inner[DType: Tensor, *Init, *Tail](
            x: TypedTensor[DType, *Init, D, *Tail], keepdim: Literal[True]
        ) -> TypedTensor[DType, *Init, Sub[D], *Tail]: ...

        @overload
        def inner[DType: Tensor, *Init, *Tail](
            x: TypedTensor[DType, *Init, D, *Tail], keepdim: Literal[False]
        ) -> TypedTensor[DType, *Init, *Tail]: ...

        def inner[DType: Tensor, *Init, *Tail](
            x: TypedTensor[DType, *Init, D, *Tail], keepdim: Literal[False] | Literal[True]
        ) -> TypedTensor[DType, *Init, *Tail] | TypedTensor[DType, *Init, Sub[D], *Tail]:
            args = list(x.args)
            dim = x.dim[tp]
            index = dim + 1
            args = args[:index] + ([Sub[tp]] if keepdim else []) + args[index + 1 :]
            return TypedTensor(cast(DType, torch.sum(x.tensor, dim=dim, keepdim=keepdim)), tuple(args))

        return inner


"""
sum[D](xs)
"""
sum = _sum()


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
