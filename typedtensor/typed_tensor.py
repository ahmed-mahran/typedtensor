from __future__ import annotations

import logging
from abc import ABCMeta
from functools import wraps
from inspect import isclass
from types import FunctionType
from typing import Any, Callable, Concatenate, Optional, Tuple, Type, TypeGuard, cast, overload, override
from mypyright_extensions import Map
import torch
from torch import Size, Tensor

from .shape_info import (
    ShapeInfo,
    _extract_typed_args,
    _extract_typed_shape_args,
)
from .utils import CaptureTypeArgs, _is_tensor_subclass, _is_type_var_of_bound

logger = logging.getLogger(__name__)


class TypedTensorMeta(ABCMeta):
    def __new__(cls, name, bases, namespace, **kwargs):
        def _check_type_error(instance: TypedTensor):
            if hasattr(instance, "_type_error") and (e := getattr(instance, "_type_error")) is not None:
                raise e

        def wrap_function[Self: TypedTensor, **P](method: Callable[Concatenate[Self, P], Any]):
            @wraps(method)
            def wrapped(instance: Self, *wargs: P.args, **wkwargs: P.kwargs):
                _check_type_error(instance)
                return method(instance, *wargs, **wkwargs)

            return wrapped

        def wrap_property[Self: TypedTensor](prop: property):
            fget = None
            if prop.fget is not None:

                def wrapped(instance: Self):
                    _check_type_error(instance)
                    if prop.fget is not None:
                        return prop.fget(instance)

                fget = wrapped

            return property(fget, prop.fset, prop.fdel, prop.__doc__)

        new_namespace = {}
        for attributeName, attribute in namespace.items():
            if isinstance(attribute, FunctionType):
                attribute = wrap_function(attribute)
            elif isinstance(attribute, property):
                attribute = wrap_property(attribute)
            new_namespace[attributeName] = attribute
        return super().__new__(cls, name, bases, new_namespace)


class TypedTensor[DType: Tensor, *Dimensions](CaptureTypeArgs, metaclass=TypedTensorMeta):
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
    _self_: Optional[TypedTensor[DType, *Dimensions]] = None
    _type_error: Optional[Exception] = None

    _args: Optional[Tuple[type, ...]] = None

    _typed_args: Optional[Tuple[Type[Tensor], ShapeInfo]] = None

    @property
    def _self(self) -> TypedTensor[DType, *Dimensions]:
        if self._type_error is not None:
            raise self._type_error
        if self._self_ is not None:
            return self._self_
        raise TypeError(f"This tensor has an invalid type: {type(self)}")

    @override
    def _on_type_args(self, type_args: Tuple[Type, ...]):
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

    def copy_with_tensor(self, tensor: DType) -> TypedTensor[DType, *Dimensions]:
        if tensor.size() != self.tensor.size():
            raise ValueError(f"Tensor sizes must match; provided {tensor.size()} while having {self.tensor.size()}")
        t = TypedTensor[DType, *Dimensions](tensor)
        t._args = self._args
        t._type_error = self._type_error
        t._typed_args = self._typed_args
        return t

    def transform(self, fn: Callable[[DType], DType]):
        return self.copy_with_tensor(fn(self.tensor))

    @property
    def typed_args(self) -> Tuple[Type[Tensor], ShapeInfo]:
        if self._typed_args is not None:
            return self._typed_args
        self._typed_args = _extract_typed_args(self._args, self.tensor)
        return self._typed_args

    @property
    def shape(self) -> ShapeInfo:
        return self.typed_args[1]

    @property
    def dim(self):
        return ShapeInfo._Dim(self.shape)

    @staticmethod
    def as_instance_of[T](t: TypedTensor[DType, *Dimensions], tp: Type[T]) -> T:
        return t.asinstanceof[tp]

    class _AsInstanceOf:
        def __init__(self, o):
            self.o = o

        def __getitem__[T](self, tp: Type[T]) -> T:
            return self.__call__(tp)

        def __call__[T](self, tp: Type[T], tp_args: Optional[Tuple[Any, ...]] = None) -> T:
            if is_instance_of(self.o, tp, tp_args):
                return self.o
            raise TypeError(f"{self.o} cannot be cast to {tp}")

    @property
    def asinstanceof(self) -> TypedTensor._AsInstanceOf:
        return TypedTensor._AsInstanceOf(self)

    class _Shaped[_DType: Tensor]:
        def __init__(self, o: TypedTensor[_DType, *Dimensions]):
            self.o = o
        
        @overload
        def __getitem__[D](self, shape: Type[D]) -> TypedTensor[_DType, D]: ...

        @overload
        def __getitem__[D, *Ds](self, shape: Map[Type, D, *Ds]) -> TypedTensor[_DType, D, *Ds]: ...

        def __getitem__[D, *Ds](self, shape: Map[Type, D, *Ds] | Type[D]) -> TypedTensor[_DType, D, *Ds] | TypedTensor[_DType, D]:
            tp = TypedTensor[_DType, D, *Ds]
            tp_args = (self.o.args[0],) + (shape if isinstance(shape, tuple) else (shape,))
            setattr(tp, "__args__", tp_args)
            return self.o.asinstanceof(tp)

    @property
    def shaped[T: Tensor](
        self: TypedTensor[T, *Dimensions],
    ) -> TypedTensor._Shaped[T]:
        return TypedTensor._Shaped[T](self)

    # class _As_Z_D0[T: Tensor]:
    #     def __init__(self, o: TypedTensor[T, *Dimensions]):
    #         self.o = o

    #     def __getitem__[D0](self, item: Type[D0]) -> TypedTensor[T, Z[Dimension], D0]:
    #         tp = TypedTensor[T, Z[Dimension], D0]
    #         tp_args = (self.o.args[0], Z[Dimension], item)
    #         setattr(tp, "__args__", tp_args)
    #         return self.o.asinstanceof(tp)

    # @property
    # def as_z_d0[T: Tensor](
    #     self: TypedTensor[T, *Dimensions],
    # ) -> TypedTensor._As_Z_D0[T]:
    #     return TypedTensor._As_Z_D0[T](self)

    # class _As_Z_D0_D1[T: Tensor]:
    #     def __init__(self, o: TypedTensor[T, *Dimensions]):
    #         self.o = o

    #     def __getitem__[D0, D1](self, item: ShapeArgs[D0, D1]) -> TypedTensor[T, Z[Dimension], D0, D1]:
    #         tp = TypedTensor[T, Z[Dimension], D0, D1]
    #         d0, d1 = Shape.types_from(item)
    #         tp_args = (self.o.args[0], Z[Dimension], d0, d1)
    #         setattr(tp, "__args__", tp_args)
    #         return self.o.asinstanceof(tp)

    # @property
    # def as_z_d0_d1[T: Tensor](
    #     self: TypedTensor[T, *Dimensions],
    # ) -> TypedTensor._As_Z_D0_D1[T]:
    #     return TypedTensor._As_Z_D0_D1[T](self)

    # class _As_Z_D0_Z[T: Tensor]:
    #     def __init__(self, o: TypedTensor[T, *Dimensions]):
    #         self.o = o

    #     def __getitem__[D0](self, item: Type[D0]) -> TypedTensor[T, Z[Dimension], D0, Z[Dimension]]:
    #         tp = TypedTensor[T, Z[Dimension], D0, Z[Dimension]]
    #         tp_args = (self.o.args[0], Z[Dimension], item, Z[Dimension])
    #         setattr(tp, "__args__", tp_args)
    #         return self.o.asinstanceof(tp)

    # @property
    # def as_z_d0_z[T: Tensor](
    #     self: TypedTensor[T, *Dimensions],
    # ) -> TypedTensor._As_Z_D0_Z[T]:
    #     return TypedTensor._As_Z_D0_Z[T](self)

    # class _As_Z_D0_Z_D1_Z[T: Tensor]:
    #     def __init__(self, o: TypedTensor[T, *Dimensions]):
    #         self.o = o

    #     def __getitem__[D0, D1](
    #         self, item: ShapeArgs[D0, D1]
    #     ) -> TypedTensor[T, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]:
    #         tp = TypedTensor[T, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]
    #         d0, d1 = Shape.types_from(item)
    #         tp_args = (self.o.args[0], Z[Dimension], d0, Z[Dimension], d1, Z[Dimension])
    #         setattr(tp, "__args__", tp_args)
    #         return self.o.asinstanceof(tp)

    # @property
    # def as_z_d0_z_d1_z[T: Tensor](
    #     self: TypedTensor[T, *Dimensions],
    # ) -> TypedTensor._As_Z_D0_Z_D1_Z[T]:
    #     return TypedTensor._As_Z_D0_Z_D1_Z[T](self)

    class _IsInstanceOf:
        def __init__(self, o):
            self.o = o

        def __getitem__[T](self, tp: Type[T]) -> TypeGuard[T]:
            return is_instance_of(self.o, tp)

    @property
    def isinstanceof(self) -> TypedTensor._IsInstanceOf:
        return TypedTensor._IsInstanceOf(self)

    @staticmethod
    def assert_valid_typed_tensor[T: Tensor, *Ds](
        tensor: TypedTensor[T, *Ds],
    ) -> TypeGuard[TypedTensor[T, *Ds]]:
        _, _ = tensor.typed_args
        return True

    def __repr__(self):
        _, shape_info = self.typed_args
        return f"{shape_info}: {self.tensor}"

    def matmul[*Ds, D0, D1, D2](
        self: TypedTensor[DType, *Ds, D0, D1],
        other: TypedTensor[DType, *Ds, D1, D2],
    ) -> TypedTensor[DType, *Ds, D0, D2]:
        args = self.args[:-1] + (other.args[-1],)
        return TypedTensor(cast(DType, self.tensor.matmul(other.tensor)), args)

    # It is currently hard to implement transpose as a method. We need a way to capture D0 and D1 before matching self
    # shape [*Init, D0, *Mid, D1, *Tail], this is currently impossible as self always comes first, hence D0 and D1
    # will be arbirarily captured from shape and not from exiplicitly passed types.
    # E.g. in transpose(self: [*Init, D0, *Mid, D1, *Tail], d0: Type[D0], d1: Type[D1]).
    # Subscriptable methods/functions is a possible solution, where caller needs to specify D0 and D1 first
    # before passing call parameters: transpose[D0, D1]()
    # 
    # class _Transpose[_DType: Tensor]:
    #     def __init__(self, o):
    #         self.o = o

    #     def __getitem__[*Init, D0, *Mid, D1, *Tail](
    #         self, shape: ShapeArgs[D0, D1]
    #     ) -> TypedTensor[_DType, *Init, D1, *Mid, D0, *Tail]:
    #         d0, d1 = Shape.types_from(shape)
    #         dim0, dim1 = tuple([self.o.dim[tp] for tp in (d0, d1)])
    #         return self(cast(Type[D0], d0), cast(Type[D1], d1), dim0, dim1)

    #     def __call__[*Init, D0, *Mid, D1, *Tail](
    #         self, d0: Type[D0], d1: Type[D1], dim0: int, dim1: int
    #     ) -> TypedTensor[_DType, *Init, D1, *Mid, D0, *Tail]:
    #         ts = list(self.o.args[1:])
    #         d0, d1 = ts[dim0], ts[dim1]  # we prefer concrete types from tensor definition
    #         ts[dim0] = d1
    #         ts[dim1] = d0
    #         return TypedTensor(cast(_DType, self.o.tensor.transpose(dim0, dim1)), tuple([self.o.args[0]] + ts))

    # @property
    # def transpose(self) -> TypedTensor._Transpose[DType]:
    #     return TypedTensor._Transpose[DType](self)
    # def transpose[*Init, D0, *Mid, D1, *Tail](
    #         self: TypedTensor[DType, *Init, D0, *Mid, D1, *Tail], d0: Type[D0], d1: Type[D1]
    # ) -> TypedTensor[DType, *Init, D1, *Mid, D0, *Tail]:
    #     dim0, dim1 = self.dim[d0], self.dim[d1]
    #     ts = list(self.args[1:])
    #     d0, d1 = ts[dim0], ts[dim1]  # we prefer concrete types from tensor definition
    #     ts[dim0] = d1
    #     ts[dim1] = d0
    #     return TypedTensor(cast(DType, self.tensor.transpose(dim0, dim1)), tuple([self.args[0]] + ts))

    class _Permute[_DType: Tensor]:
        def __init__(self, o: TypedTensor[_DType, *Dimensions]):
            self.o = o

        @overload
        def __getitem__[P](self, shape: Type[P]) -> TypedTensor[_DType, P]: ...

        @overload
        def __getitem__[P, *Ps](self, shape: Map[Type, P, *Ps]) -> TypedTensor[_DType, P, *Ps]: ...

        def __getitem__[P, *Ps](self, shape: Map[Type, P, *Ps] | Type[P]) -> TypedTensor[_DType, P, *Ps] | TypedTensor[_DType, P]:
            types = shape if isinstance(shape, tuple) else (shape,)
            dims = [self.o.dim[tp] for tp in types]
            return TypedTensor(cast(_DType, self.o.tensor.permute(dims)), (self.o.args[0],) + types)

    @property
    def permute(self):
        return TypedTensor._Permute[DType](self)

    class _View[_DType: Tensor]:
        def __init__(self, o):
            self.o = o

        @overload
        def __getitem__[V](self, shape: Type[V]) -> Callable[[Optional[Size]], TypedTensor[_DType, V]]: ...

        @overload
        def __getitem__[V, *Vs](self, shape: Map[Type, V, *Vs]) -> Callable[[Optional[Size]], TypedTensor[_DType, V, *Vs]]: ...

        def __getitem__[V, *Vs](self, shape: Map[Type, V, *Vs] | Type[V]) -> Callable[[Optional[Size]], TypedTensor[_DType, V, *Vs]] | Callable[[Optional[Size]], TypedTensor[_DType, V]]:
            if isinstance(shape, tuple):
                def innerVVs(size: Optional[Size] = None) -> TypedTensor[_DType, V, *Vs]:
                    return self(shape, size)

                return innerVVs
            else:
                def innerV(size: Optional[Size] = None) -> TypedTensor[_DType, V]:
                    return self((shape,), size)

                return innerV
        # def __getitem__[*Vs](self, shape: Map[Type, *Vs]) -> Callable[[Optional[Size]], TypedTensor[_DType, *Vs]]:
        #     def innerVs(size: Optional[Size] = None) -> TypedTensor[_DType, *Vs]:
        #         # return self(shape if isinstance(shape, tuple) else (shape,), size)
        #         return self(shape, size)

        #    return innerVs

        def __call__[*Ks](
            self,
            shape: Map[Type, *Ks],
            size: Optional[Size] = None,
        ) -> TypedTensor[_DType, *Ks]:
            if size is None:
                shape_info = _extract_typed_shape_args(shape)
                size = shape_info.size()
                if sum([1 if s < 0 else 0 for s in size]) > 1:
                    raise TypeError("At most one dimension can be abstract or otherwise provide size argument")
            return TypedTensor(cast(_DType, self.o.tensor.view(size)), (self.o.args[0],) + shape)


    @property
    def view(self):
        return TypedTensor._View[DType](self)

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

    def contiguous(self, *args, **kwargs):
        return self.copy_with_tensor(self.tensor.contiguous(*args, **kwargs))

    def __add__(self, other: DType):
        return self.transform(lambda t: cast(DType, t + other))

    def __mul__(self, other: DType):
        return self.transform(lambda t: cast(DType, t * other))

    def __truediv__(self, other: DType | int | float):
        return self.transform(lambda t: cast(DType, t / other))


def is_instance_of[DType: Tensor, *Dimensions, T](
    t: TypedTensor[DType, *Dimensions], tp: Type[T], tp_args: Optional[Tuple[Any, ...]] = None
) -> TypeGuard[T]:
    tensor_dtype, tensor_shape_info = t.typed_args
    if tp_args is None and hasattr(tp, "__args__"):
        tp_args = getattr(tp, "__args__")

    if tp_args is not None:
        type_dtype, type_shape_info = _extract_typed_args(tp_args)
    else:
        return False

    if isclass(type_dtype):
        if not _is_tensor_subclass(tensor_dtype, type_dtype):
            return False
    elif _is_type_var_of_bound(type_dtype, Tensor):
        if type_dtype.__bound__ is None or not _is_tensor_subclass(tensor_dtype, type_dtype.__bound__):
            return False

    logger.debug(f"Is {tensor_shape_info.args} <= {type_shape_info.args}?")

    return tensor_shape_info.matches(type_shape_info)
