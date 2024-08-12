from __future__ import annotations

import logging
import math
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import isclass
from typing import Any, List, Optional, Tuple, Type, TypeGuard, TypeVarTuple, overload

from torch import Size, Tensor

from .dimension import Concat, Dimension, Rec, Z
from .utils import (
    CapturedTypeArgs,
    _is_generic_type,
    _is_tensor_subclass,
    _is_type_var_of_bound,
    _Unpack_type,
    get_common_sequence,
    match_sequence,
)

logger = logging.getLogger(__name__)

####################
# Dimension Length #
####################


class DimensionLength(ABC):
    @property
    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    @abstractmethod
    def __add__(self, other):
        pass


@dataclass
class ExactDimensionLength(DimensionLength):
    value: int

    @property
    def length(self) -> int:
        return self.value

    def __eq__(self, other):
        return isinstance(other, ExactDimensionLength) and self.length == other.length

    def __lt__(self, other):
        return isinstance(other, ExactDimensionLength) and self.length < other.length

    def __add__(self, other):
        if isinstance(other, ExactDimensionLength):
            return ExactDimensionLength(value=self.length + other.length)
        elif isinstance(other, UnboundDimensionLength):
            return AtLeastDimensionLength(min_threshold=self.length)
        elif isinstance(other, AtLeastDimensionLength):
            return AtLeastDimensionLength(min_threshold=self.length + other.min_threshold)
        raise NotImplementedError(f"Unrecognized argument {other} of type {type(other)}")

    def __repr__(self):
        return str(self.value)


class UnboundDimensionLength(DimensionLength):
    @property
    def length(self) -> int:
        return math.inf  # type: ignore

    def __eq__(self, other):
        return isinstance(other, UnboundDimensionLength) or isinstance(other, AtLeastDimensionLength)

    def __lt__(self, other):
        return False

    def __add__(self, other):
        if isinstance(other, ExactDimensionLength):
            return AtLeastDimensionLength(min_threshold=other.length)
        elif isinstance(other, UnboundDimensionLength):
            return UnboundDimensionLength()
        elif isinstance(other, AtLeastDimensionLength):
            return AtLeastDimensionLength(min_threshold=other.min_threshold)
        raise NotImplementedError(f"Unrecognized argument {other} of type {type(other)}")

    def __repr__(self):
        return "_"


@dataclass
class AtLeastDimensionLength(DimensionLength):
    min_threshold: int

    @property
    def length(self) -> int:
        return math.inf  # type: ignore

    def __eq__(self, other):
        return isinstance(other, UnboundDimensionLength) or isinstance(other, AtLeastDimensionLength)

    def __lt__(self, other):
        return False

    def __add__(self, other):
        if isinstance(other, ExactDimensionLength):
            return AtLeastDimensionLength(min_threshold=self.min_threshold + other.length)
        elif isinstance(other, UnboundDimensionLength):
            return AtLeastDimensionLength(min_threshold=self.min_threshold)
        elif isinstance(other, AtLeastDimensionLength):
            return AtLeastDimensionLength(min_threshold=self.min_threshold + other.min_threshold)
        raise NotImplementedError(f"Unrecognized argument {other} of type {type(other)}")

    def __repr__(self):
        return f"_>={self.min_threshold}"


######################
# Dimension Arg Info #
######################


class DimensionArgInfo(ABC):
    @property
    @abstractmethod
    def origin(self) -> Type:
        pass

    @property
    @abstractmethod
    def length(self) -> DimensionLength:
        pass

    @abstractmethod
    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class NestableDimensionArgInfo(DimensionArgInfo, ABC):
    pass


class FunctorDimensionArgInfo(NestableDimensionArgInfo, ABC):
    pass


class ConcreteDimensionArgInfo(NestableDimensionArgInfo):
    def __init__(self, name: str, length: int, origin: Type[Dimension]):
        self.name = name
        self._length = length
        self._origin = origin

    @property
    def origin(self) -> Type:
        return self._origin

    @property
    def length(self):
        return ExactDimensionLength(value=self._length)

    def __eq__(self, other):
        return (
            isinstance(other, ConcreteDimensionArgInfo)
            and self.name == other.name
            and self._length == other._length
            and self.origin == other.origin
        )

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return issubclass(self.origin, parent.origin) and self.length == parent.length
        elif isinstance(parent, AbstractDimensionArgInfo):
            return issubclass(self.origin, parent.origin)
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        elif isinstance(parent, ConcatDimensionArgInfo):
            return False
        elif isinstance(parent, RecDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"{self.name}({self.length})"


class AbstractDimensionArgInfo(NestableDimensionArgInfo):
    def __init__(self, name: str, origin: Type[Dimension]) -> None:
        self.name = name
        self._origin = origin

    @property
    def origin(self) -> Type:
        return self._origin

    @property
    def length(self):
        return UnboundDimensionLength()

    def __eq__(self, other):
        return isinstance(other, AbstractDimensionArgInfo) and self.name == other.name and self.origin == other.origin

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return False
        elif isinstance(parent, AbstractDimensionArgInfo):
            return issubclass(self.origin, parent.origin)
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        elif isinstance(parent, ConcatDimensionArgInfo):
            return self.is_subclass(parent.left) and self.is_subclass(parent.right)
        elif isinstance(parent, RecDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"{self.name}(_)"


class UnboundAbstractDimensionArgInfo(NestableDimensionArgInfo):
    def __init__(self, name: str = "Dimension", origin: Any = None) -> None:
        self.name = name
        self._origin = origin

    @property
    def origin(self) -> Type:
        return self._origin

    @property
    def length(self):
        return UnboundDimensionLength()

    def __eq__(self, other):
        return isinstance(other, UnboundAbstractDimensionArgInfo) and self.name == other.name

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return False
        elif isinstance(parent, AbstractDimensionArgInfo):
            return False
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        elif isinstance(parent, ConcatDimensionArgInfo):
            return self.is_subclass(parent.left) and self.is_subclass(parent.right)
        elif isinstance(parent, RecDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"{self.name}(_)"


Unkown = UnboundAbstractDimensionArgInfo(name="Unkown")


class RepeatedDimensionArgInfo(DimensionArgInfo):
    def __init__(self, base: NestableDimensionArgInfo, origin: Any) -> None:
        self.base = base
        self._origin = origin

    @property
    def origin(self) -> Type:
        return self._origin

    @property
    def length(self):
        return self.base.length

    def __eq__(self, other):
        return isinstance(other, RepeatedDimensionArgInfo) and self.base == other.base

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        return self.base.is_subclass(parent)

    def __repr__(self):
        return f"{self.base}*"


class ConcatDimensionArgInfo(FunctorDimensionArgInfo):
    def __init__(self, left: NestableDimensionArgInfo, right: NestableDimensionArgInfo, origin: Any) -> None:
        self.left = left
        self.right = right
        self._origin = origin

    @property
    def origin(self) -> Type:
        return self._origin

    @property
    def length(self):
        return self.left.length + self.right.length

    def __eq__(self, other):
        return isinstance(other, ConcatDimensionArgInfo) and self.length == other.length and self.right == other.right

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return False
        elif isinstance(parent, AbstractDimensionArgInfo):
            return False
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        elif isinstance(parent, ConcatDimensionArgInfo):
            return self.left.is_subclass(parent.left) and self.right.is_subclass(parent.right)
        elif isinstance(parent, RecDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"Concat[{self.left}, {self.right}]"


class RecDimensionArgInfo(FunctorDimensionArgInfo):
    def __init__(self, base: NestableDimensionArgInfo, func: FunctorDimensionArgInfo, origin: Any) -> None:
        self.base = base
        self.func = func
        self._origin = origin

    @property
    def origin(self) -> Type:
        return self._origin

    @property
    def length(self):
        return self.base.length + self.func.length

    def __eq__(self, other):
        return isinstance(other, RecDimensionArgInfo) and self.base == other.base and self.func == other.func

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, RecDimensionArgInfo):
            return self.func.is_subclass(parent.func) and self.base.is_subclass(parent.base)
        return self.base.is_subclass(parent)

    def __repr__(self):
        return f"Rec[{self.base}, {self.func}]"


class Shape[*Ds](Dimension, CapturedTypeArgs):
    @staticmethod
    def types_from(shape: ShapeArgs[*Ds]) -> Tuple[Type[Dimension], ...]:
        tps = getattr(shape, "__args__")
        return tuple([tp for tp in tps if isclass(tp) and issubclass(tp, Dimension)])

    @property
    def shape_info(self):
        return ShapeInfo(_unpack_recognize_args(self.type_args))


type ShapeArgs[*Ds] = Type[Shape[*Ds]]


class Broadcast[A: Shape, B: Shape](Dimension, CapturedTypeArgs):
    @staticmethod
    def broadcast(a: List[DimensionArgInfo], b: List[DimensionArgInfo]) -> List[DimensionArgInfo]:
        """
        Broadcast semantics on type level are defined differently. Shapes are aligned from right
        to left. Dimension types from the higher dimensional shape on the left that don't align
        with any dimension type from the lower dimensional shape are retruned as-is. For each
        pair of the aligned dimension types, the type with longer length is returned given that
        the other has length 1, or otherwise the type which is super to the other type is returned,
        or otherwise broadcasting fails.

        Currently broadcasting doesn't handle repeated dimensions. This is because repeated dimensions
        shouldn't be used at runtime. However, if this logic to be run at static type checking time,
        repeated dimensions must be handled somehow.
        """

        def get_common[I: DimensionArgInfo](a: Optional[I], b: Optional[I]):
            if a is not None and b is not None:
                if a.length > b.length and b.length.length == 1:
                    return a
                elif b.length > a.length and a.length.length == 1:
                    return b
                elif a.is_subclass(b):
                    return b
                elif b.is_subclass(a):
                    return a
            elif a is not None and b is None:
                return a
            elif a is None and b is not None:
                return b
            return None

        def verify_not_repeated(args: List[DimensionArgInfo]):
            for arg in args:
                if _is_repeated(arg):
                    raise TypeError(f"Cannot determine broadcast semantics of repeated dimensions {arg}")

        verify_not_repeated(a)
        verify_not_repeated(b)
        long, short = (a, b) if len(a) >= len(b) else (b, a)
        long_shape = [i for i in long[len(long) - len(short) :]]
        short_shape = [i for i in short]
        common = get_common_sequence(
            long_shape, short_shape, _is_repeated, _is_repeated, _is_repeated, get_common, logger
        )

        if common is not None:
            return long[: len(long) - len(short)] + common
        # if long_shape.matches(short_shape):
        #     return long_shape.args
        raise TypeError(f"{ShapeInfo(a)} and {ShapeInfo(b)} are not broadcastable")


@dataclass
class ShapeInfo:
    args: List[DimensionArgInfo]

    def types(self):
        return [a.origin for a in self.args]

    class _Dim:
        def __init__(self, o):
            self.o = o

        def __getitem__[T: Dimension](self, tp: Type[T]):
            arg = _unpack_recognize_arg(tp)[0]
            for i, item in enumerate(self.o.args):
                if item.is_subclass(arg):
                    return i
            raise ValueError(f"Dimension {tp} doesn't exist in shape {self.o}")

    @property
    def dim(self):
        return ShapeInfo._Dim(self)

    def size(self) -> Size:
        return Size([a.value if isinstance(a, ExactDimensionLength) else -1 for a in self.args])

    def _matches_shapeinfo(self, other: ShapeInfo) -> bool:
        def a_matches_b[I: DimensionArgInfo](a: I, b: I) -> bool:
            return a.is_subclass(b)

        return match_sequence(self.args, other.args, _is_repeated, _is_repeated, a_matches_b, logger)

    def _matches_size(self, size: Size) -> bool:
        def a_matches_b(a: DimensionArgInfo, b: int) -> bool:
            if isinstance(a, ConcreteDimensionArgInfo):
                return a.length == ExactDimensionLength(b)
            elif isinstance(a, AbstractDimensionArgInfo):
                return True
            elif isinstance(a, UnboundAbstractDimensionArgInfo):
                return True
            elif isinstance(a, RepeatedDimensionArgInfo):
                return a_matches_b(a.base, b)
            elif isinstance(a, ConcatDimensionArgInfo):
                a_length = a.length
                if isinstance(a_length, ExactDimensionLength):
                    return a_length == ExactDimensionLength(b)
                elif isinstance(a_length, UnboundDimensionLength):
                    return True
                elif isinstance(a_length, AtLeastDimensionLength):
                    return a_length >= ExactDimensionLength(b)
            elif isinstance(a, RecDimensionArgInfo):
                a_length = a.length
                if isinstance(a_length, ExactDimensionLength):
                    return a_length == ExactDimensionLength(b)
                elif isinstance(a_length, UnboundDimensionLength):
                    return True
                elif isinstance(a_length, AtLeastDimensionLength):
                    return a_length >= ExactDimensionLength(b)
            raise TypeError(f"Unrecognized dimension arg {a}")

        return match_sequence(self.args, list(size), _is_repeated, lambda _: False, a_matches_b, logger)

    @overload
    def matches(self, other: Size) -> bool: ...

    @overload
    def matches(self, other: ShapeInfo) -> bool: ...

    def matches(self, other: Size | ShapeInfo) -> bool:
        if isinstance(other, Size):
            return self._matches_size(other)
        else:
            return self._matches_shapeinfo(other)

    def __repr__(self):
        return " x ".join(map(str, self.args))


def _unpack_recognize_arg(arg: Any) -> List[DimensionArgInfo]:
    # [..., arg = Dimension, ...]
    if arg is Dimension:
        return [AbstractDimensionArgInfo(name="Dimension", origin=arg)]
    # [..., arg = Concat, ...]
    elif arg is Concat:
        return [ConcatDimensionArgInfo(left=Unkown, right=Unkown, origin=arg)]
    # [..., arg = Shape[*Ds], ...]
    elif _is_generic_type(arg, Shape):
        return _unpack_recognize_args(getattr(arg, "__args__"))
    # [..., arg = Broadcast[*Ds], ...]
    elif _is_generic_type(arg, Broadcast):
        args = getattr(arg, "__args__")
        a = _unpack_recognize_arg(args[0])
        b = _unpack_recognize_arg(args[1])
        return Broadcast.broadcast(a, b)
    # [..., arg <= Dimension, ...]
    elif isclass(arg) and issubclass(arg, Dimension):
        # [..., arg <= Dimension(length=...), ...]
        if hasattr(arg, "length"):
            return [ConcreteDimensionArgInfo(name=arg.__name__, length=arg.length, origin=arg)]
        else:
            return [AbstractDimensionArgInfo(name=arg.__name__, origin=arg)]
    # [..., arg = T: bound <= Dimension, ...]
    elif _is_type_var_of_bound(arg, Dimension) and arg.__bound__ is not None:
        return [AbstractDimensionArgInfo(name=arg.__name__, origin=arg.__bound__)]
    # [..., arg = T: bound = None, ...]
    elif _is_type_var_of_bound(arg, None):
        return [UnboundAbstractDimensionArgInfo(name=arg.__name__, origin=arg)]
    # [..., arg = Concat[L, R], ...]
    elif _is_generic_type(arg, Concat):
        concat_args = getattr(arg, "__args__")
        left = _unpack_recognize_arg(concat_args[0])[0]
        right = _unpack_recognize_arg(concat_args[1])[0]
        if isinstance(left, NestableDimensionArgInfo) and isinstance(right, NestableDimensionArgInfo):
            return [ConcatDimensionArgInfo(left=left, right=right, origin=arg)]
    # [..., arg = Rec[A, T], ...]
    elif _is_generic_type(arg, Rec):
        rec_args = getattr(arg, "__args__")
        base = _unpack_recognize_arg(rec_args[0])[0]
        func = _unpack_recognize_arg(rec_args[1])[0]
        if isinstance(base, NestableDimensionArgInfo) and isinstance(func, FunctorDimensionArgInfo):
            return [RecDimensionArgInfo(base=base, func=func, origin=arg)]
    # [..., arg = Z[T], ...]
    elif _is_generic_type(arg, Z):
        # base = T = arg.__args__[0]
        base = _unpack_recognize_arg(getattr(arg, "__args__")[0])[0]
        if isinstance(base, NestableDimensionArgInfo):
            return [RepeatedDimensionArgInfo(base, origin=arg)]
    # [..., arg = *Ts | *Tuple[...], ...]
    elif issubclass(type(arg), _Unpack_type) or issubclass(type(arg), types.GenericAlias):
        unpacked = getattr(arg, "__args__")[0] if issubclass(type(arg), _Unpack_type) else arg
        if isinstance(unpacked, TypeVarTuple):
            return [RepeatedDimensionArgInfo(UnboundAbstractDimensionArgInfo(origin=unpacked), origin=arg)]
        # unpacked is Tuple[T, ...] | Tuple[A, B, ...]
        if hasattr(unpacked, "__origin__") and getattr(unpacked, "__origin__") is tuple:
            # unpacked is Tuple[T, ...] i.e. T zero or more
            if len(getattr(unpacked, "__args__")) == 2 and getattr(unpacked, "__args__")[1] is ...:
                # base = T
                base = _unpack_recognize_arg(getattr(unpacked, "__args__")[0])[0]
                if isinstance(base, NestableDimensionArgInfo):
                    return [RepeatedDimensionArgInfo(base, origin=arg)]
            # unpacked is Tuple[A, B, ...] i.e. a bounded tuple of types
            return _unpack_recognize_args(getattr(unpacked, "__args__"))
    raise TypeError(f"Unrecognized shape parameter {arg} of type {type(arg)}")


def _extract_typed_args[DType: Tensor](
    _args: Optional[Tuple[Any, ...]], tensor: Optional[DType] = None
) -> Tuple[Type[Tensor], ShapeInfo]:
    if _args is None or len(_args) == 0:
        raise TypeError("Cannot verify type of tensor; TypedTensor[] class has no arguments")

    all_args = _args
    if len(all_args) < 1:
        raise TypeError("Type arguments not enough, provided %s" % (str(all_args)))

    arg_0 = all_args[0]

    d_type: Type[Tensor]
    if _is_type_var_of_bound(arg_0, Tensor):
        if tensor is not None and isinstance(tensor, Tensor) and tensor.__class__ is not None:
            d_type = tensor.__class__
        elif tensor is None and arg_0.__bound__ is not None:
            d_type = arg_0.__bound__
    elif isclass(arg_0) and _is_tensor_subclass(arg_0, Tensor):
        d_type = arg_0
    else:
        raise TypeError(f"TypedTensor data type must be <= torch.Tensor but got {arg_0} of type {type(arg_0)}")

    shape_info = _extract_typed_shape_args(all_args[1:])

    if tensor is not None:
        if not shape_info.matches(tensor.size()):
            raise TypeError(f"Tensor size {tensor.size()} did not match shape arguments {shape_info}")

    return d_type, shape_info


def _unpack_recognize_args(_args: Tuple[Any, ...]) -> List[DimensionArgInfo]:
    shape_args: List[DimensionArgInfo] = []
    for arg in _args:
        shape_args.extend(_unpack_recognize_arg(arg))

    return shape_args


def _extract_typed_shape_args(_args: Optional[Tuple[Any, ...]]) -> ShapeInfo:
    if _args is None or len(_args) == 0:
        raise TypeError("Cannot verify shape of tensor; no arguments provided")

    return ShapeInfo(_unpack_recognize_args(_args))


def _is_repeated(
    arg: Optional[DimensionArgInfo],
) -> TypeGuard[RepeatedDimensionArgInfo]:
    return arg is not None and isinstance(arg, RepeatedDimensionArgInfo)
