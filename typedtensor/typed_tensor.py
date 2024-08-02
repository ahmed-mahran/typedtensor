from __future__ import annotations

import logging
from inspect import isclass
from typing import Callable, Optional, Tuple, Type, TypeGuard, cast, overload, override

import torch
from torch import Size, Tensor

from .dimension import Dimension, Z
from .shape_info import DimensionArgInfo, ShapeInfo, _extract_typed_args, _is_repeated
from .utils import CaptureTypeArgs, _is_tensor_subclass, _is_type_var_of_bound, match_sequence

logger = logging.getLogger(__name__)


class TypedTensor[DType: Tensor, *Dimensions](CaptureTypeArgs):
    # When TypedTensor class is subscripted, a new _GenericAlias
    # is created which holds three special attributes for internal bookkeeping of generic types:
    # * __parameters__ is a tuple of unique free type parameters of a generic
    #   type, for example, TypedTensor[T, T].__parameters__ == (T,);
    # * __origin__ keeps a reference to a type that was subscripted,
    #   e.g., TypedTensor[T, int].__origin__ == TypedTensor, or the
    #   non-generic version of the type.
    # * __args__ is a tuple of all arguments used in subscripting,
    #   e.g., TypedTensor[T, int].__args__ == (T, int).
    #
    # Then, when its base class _BaseGenericAlias is called (__call__ method),
    # it creates an instance from TypedTensor and sets the __orig_class__
    # attribute with a value of itself (the _GenericAlias instance).
    #
    # __orig_class__ attribute on any TypedTensor instance is not available
    # during __init__ method
    _self_: Optional[TypedTensor[DType, Z[Dimension]]] = None
    _type_error: Optional[Exception] = None

    _args: Optional[Tuple[type, ...]] = None

    _typed_args: Optional[Tuple[Type[Tensor], ShapeInfo]] = None

    @property
    def _self(self) -> TypedTensor[DType, Z[Dimension]]:
        if self._type_error is not None:
            raise self._type_error
        if self._self_ is not None:
            return self._self_
        raise TypeError(f"This tensor has an invalid type: {type(self)}")

    @override
    def _on_type_args(self, type_args: Tuple[Type]):
        if self._args is None:
            self._args = type_args
            self._typed_args = None
            try:
                if TypedTensor.assert_valid_typed_tensor(self):
                    self._self_ = self
            except Exception as type_error:
                self._type_error = type_error

    @property
    def args(self) -> Tuple[type, ...]:
        if self._args is not None:
            return self._args
        raise TypeError("TypedTensor has no type args")

    @args.setter
    def args(self, value):
        self._args = value
        self._typed_args = None
        if TypedTensor.assert_valid_typed_tensor(self):
            self._self_ = self

    def __init__(self, tensor: DType, t_args: Optional[Tuple[type, ...]] = None):
        self.tensor = tensor
        if t_args is not None:
            self._args = t_args
            self._typed_args = None
            if TypedTensor.assert_valid_typed_tensor(self):
                self._self_ = self

    def copy_with_tensor[T: TypedTensor](self: T, tensor: DType) -> T:
        if tensor.size() != self.tensor.size():
            raise ValueError(f"Tensor sizes must match; provided {tensor.size()} while having {self.tensor.size()}")
        t = TypedTensor(tensor)
        t._args = self._args
        t._type_error = self._type_error
        t._typed_args = self._typed_args
        return cast(T, t)

    def transform(self, fn: Callable[[DType], DType]):
        return self.copy_with_tensor(fn(self.tensor))

    @property
    def typed_args(self) -> Tuple[Type[Tensor], ShapeInfo]:
        if self._typed_args is not None:
            return self._typed_args
        self._typed_args = _extract_typed_args(self._args, self.tensor)
        return self._typed_args

    @staticmethod
    def as_instance_of[T](t: TypedTensor[DType, *Dimensions], tp: Type[T]) -> T:
        return t.asinstanceof[tp]

    class _AsInstanceOf:
        def __init__(self, o):
            self.o = o

        def __getitem__[T](self, item: Type[T]) -> T:
            if is_instance_of(self.o, item):
                return self.o
            raise TypeError(f"{self.o} cannot be cast to {item}")

    @property
    def asinstanceof(self) -> TypedTensor._AsInstanceOf:
        return TypedTensor._AsInstanceOf(self)

    class _As_Z_D0[T: Tensor]:
        def __init__(self, o: TypedTensor[T, *Dimensions]):
            self.o = o

        def __getitem__[D0](self, item: Type[D0]) -> TypedTensor[T, Z[Dimension], D0]:
            return self.o.asinstanceof[TypedTensor[T, Z[Dimension], D0]]

    @property
    def as_z_d0[T: Tensor](
        self: TypedTensor[T, *Dimensions],
    ) -> TypedTensor._As_Z_D0[T]:
        return TypedTensor._As_Z_D0[T](self)

    class _As_Z_D0_D1[T: Tensor]:
        def __init__(self, o: TypedTensor[T, *Dimensions]):
            self.o = o

        def __getitem__[D0, D1](self, item: Tuple[D0, D1]) -> TypedTensor[T, Z[Dimension], D0, D1]:
            return self.o.asinstanceof[TypedTensor[T, Z[Dimension], D0, D1]]

    @property
    def as_z_d0_d1[T: Tensor](
        self: TypedTensor[T, *Dimensions],
    ) -> TypedTensor._As_Z_D0_D1[T]:
        return TypedTensor._As_Z_D0_D1[T](self)

    class _As_Z_D0_Z_D1_Z[T: Tensor]:
        def __init__(self, o: TypedTensor[T, *Dimensions]):
            self.o = o

        def __getitem__[D0, D1](
            self, item: Tuple[D0, D1]
        ) -> TypedTensor[T, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]:
            return self.o.asinstanceof[TypedTensor[T, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]]

    @property
    def as_z_d0_z_d1_z[T: Tensor](
        self: TypedTensor[T, *Dimensions],
    ) -> TypedTensor._As_Z_D0_Z_D1_Z[T]:
        return TypedTensor._As_Z_D0_Z_D1_Z[T](self)

    class _IsInstanceOf:
        def __init__(self, o):
            self.o = o

        def __getitem__[T](self, item: Type[T]) -> TypeGuard[T]:
            return is_instance_of(self.o, item)

    @property
    def isinstanceof(self) -> TypedTensor._IsInstanceOf:
        return TypedTensor._IsInstanceOf(self)

    @staticmethod
    def assert_valid_typed_tensor[T: Tensor, *Ds](
        tensor: TypedTensor[T, *Ds],
    ) -> TypeGuard[TypedTensor[T, Z[Dimension]]]:
        _, _ = tensor.typed_args
        return True

    @staticmethod
    def is_at_least_2d[D0, D1](
        tensor: TypedTensor[DType, Z[Dimension]],
        d0: Optional[Type[D0]] = None,
        d1: Optional[Type[D1]] = None,
    ) -> TypeGuard[TypedTensor[DType, Z[Dimension], D0, D1]]:
        return tensor.isinstanceof[TypedTensor[DType, Z[Dimension], D0, D1]]

    def __repr__(self):
        _, shape_info = self.typed_args
        return f"{shape_info}: {self.tensor}"

    def matmul[D0, D1, D2](
        self: TypedTensor[DType, Z[Dimension], D0, D1],
        other: TypedTensor[DType, Z[Dimension], D1, D2],
    ) -> TypedTensor[DType, Z[Dimension], D0, D2]:
        first = self.asinstanceof[TypedTensor[DType, Z[Dimension], D0, D1]]
        second = other.asinstanceof[TypedTensor[DType, Z[Dimension], D1, D2]]
        d2 = second.args[-1]
        ts = first.args[:-1] + (d2,)
        return TypedTensor(cast(DType, first.tensor.matmul(second.tensor)), ts)
        # first = self._self
        # if TypedTensor.is_at_least_2d(first):
        #     d1 = first.args[-2]
        #     d2 = other.args[-1]
        #     return TypedTensor[DType, Z[Dimension], d1, d2](first.tensor.matmul(other.tensor))
        # raise TypeError("matmul must be called from at least a 2D TypedTensor")

    def transpose[D0, D1](
        self: TypedTensor[DType, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]],
        dim0: int,
        dim1: int,
    ) -> TypedTensor[DType, Z[Dimension], D1, Z[Dimension], D0, Z[Dimension]]:
        me = self.asinstanceof[TypedTensor[DType, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]]
        ts = list(me.args[1:])
        d0, d1 = ts[dim0], ts[dim1]
        ts[dim0] = d1
        ts[dim1] = d0
        return TypedTensor(cast(DType, me.tensor.transpose(dim0, dim1)), tuple([me.args[0]] + ts))

    @overload
    def size(self, dim: None = None) -> Size: ...

    @overload
    def size(self, dim: int) -> int: ...

    def size(self, dim: Optional[int] = None) -> Size | int:
        return self.tensor.size(dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, *args, **kwargs):
        return self.copy_with_tensor(self.tensor.to(*args, **kwargs))

    def __add__(self, other: DType):
        return self.transform(lambda t: cast(DType, t + other))

    def __mul__(self, other: DType):
        return self.transform(lambda t: cast(DType, t * other))


def is_instance_of[DType: Tensor, *Dimensions, T](t: TypedTensor[DType, *Dimensions], tp: Type[T]) -> TypeGuard[T]:
    tensor_dtype, tensor_shape_info = t.typed_args
    if hasattr(tp, "__args__"):
        type_dtype, type_shape_info = _extract_typed_args(getattr(tp, "__args__"))
    else:
        return False

    if isclass(type_dtype):
        if not _is_tensor_subclass(tensor_dtype, type_dtype):
            return False
    elif _is_type_var_of_bound(type_dtype, Tensor):
        if type_dtype.__bound__ is None or not _is_tensor_subclass(tensor_dtype, type_dtype.__bound__):
            return False

    tensor_type_args = tensor_shape_info.args
    type_type_args = type_shape_info.args

    logger.debug(f"Is {tensor_type_args} <= {type_type_args}?")

    def a_matches_b[I: DimensionArgInfo](a: I, b: I) -> bool:
        return a.is_subclass(b)

    return match_sequence(
        0, 0, tensor_type_args, type_type_args, _is_repeated, _is_repeated, a_matches_b, "", [], [], logger
    )


def addmm[DType: Tensor, *Ds, D0, D1, D2](
    input: torch.Tensor, mat1: TypedTensor[DType, *Ds, D0, D1], mat2: TypedTensor[DType, *Ds, D1, D2]
) -> TypedTensor[DType, *Ds, D0, D2]:
    ts1 = list(mat1.args)
    ts2 = list(mat2.args)
    ts = ts1[:-3] + [ts1[-2], ts2[-1]]
    res = cast(DType, torch.addmm(input, mat1.tensor, mat2.tensor))
    return TypedTensor(res, tuple(ts))
