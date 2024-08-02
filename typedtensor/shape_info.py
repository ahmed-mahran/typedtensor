from __future__ import annotations

import logging
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import isclass
from typing import Any, List, Optional, Tuple, Type, TypeGuard, TypeVarTuple, Unpack

from torch import Size, Tensor

from .dimension import Dimension, Z
from .utils import _is_tensor_subclass, _is_type_var_of_bound, match_sequence

logger = logging.getLogger(__name__)


class DimensionArgInfo(ABC):
    @abstractmethod
    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        pass


@dataclass
class ConcreteDimensionArgInfo(DimensionArgInfo):
    name: str
    length: int
    origin: Type[Dimension]

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return issubclass(self.origin, parent.origin) and self.length == parent.length
        elif isinstance(parent, AbstractDimensionArgInfo):
            return issubclass(self.origin, parent.origin)
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"{self.name}({self.length})"


@dataclass
class AbstractDimensionArgInfo(DimensionArgInfo):
    name: str
    origin: Type[Dimension]

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return False
        elif isinstance(parent, AbstractDimensionArgInfo):
            return issubclass(self.origin, parent.origin)
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"{self.name}(_)"


@dataclass
class UnboundAbstractDimensionArgInfo(DimensionArgInfo):
    name: str = "Dimension"

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        if isinstance(parent, ConcreteDimensionArgInfo):
            return False
        elif isinstance(parent, AbstractDimensionArgInfo):
            return False
        elif isinstance(parent, UnboundAbstractDimensionArgInfo):
            return True
        elif isinstance(parent, RepeatedDimensionArgInfo):
            return self.is_subclass(parent.base)
        raise TypeError(f"Type {type(parent)} is not handled!")

    def __repr__(self):
        return f"{self.name}(_)"


@dataclass
class RepeatedDimensionArgInfo(DimensionArgInfo):
    base: ConcreteDimensionArgInfo | AbstractDimensionArgInfo | UnboundAbstractDimensionArgInfo

    def is_subclass(self, parent: DimensionArgInfo) -> bool:
        return self.base.is_subclass(parent)

    def __repr__(self):
        return f"{self.base}*"


@dataclass
class ShapeInfo:
    args: List[DimensionArgInfo]

    def matches(self, size: Size) -> bool:
        def a_matches_b(a: DimensionArgInfo, b: int) -> bool:
            if isinstance(a, ConcreteDimensionArgInfo):
                return a.length == b
            elif isinstance(a, AbstractDimensionArgInfo):
                return True
            elif isinstance(a, UnboundAbstractDimensionArgInfo):
                return True
            elif isinstance(a, RepeatedDimensionArgInfo):
                return a_matches_b(a.base, b)
            raise TypeError(f"Unrecognized dimension arg {a}")

        return match_sequence(self.args, list(size), _is_repeated, lambda _: False, a_matches_b, logger)

    def __repr__(self):
        return " x ".join(map(str, self.args))


# typing._UnpackGenericAlias
_Unpack_type = type(Unpack[...])  # type: ignore


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

    def _unpack_recognize_arg(arg: Any) -> List[DimensionArgInfo]:
        # [..., arg = Dimension, ...]
        if arg == Dimension:
            return [AbstractDimensionArgInfo(name="Dimension", origin=Dimension)]
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
            return [UnboundAbstractDimensionArgInfo(name=arg.__name__)]
        # [..., arg = Z[T], ...]
        elif (
            issubclass(type(arg), type(Z[Any]))
            and hasattr(arg, "__origin__")
            and getattr(arg, "__origin__") is Z
            and hasattr(arg, "__args__")
        ):
            # base = T = arg.__args__[0]
            base = _unpack_recognize_arg(getattr(arg, "__args__")[0])[0]
            if isinstance(
                base,
                ConcreteDimensionArgInfo | AbstractDimensionArgInfo | UnboundAbstractDimensionArgInfo,
            ):
                return [RepeatedDimensionArgInfo(base)]
        # [..., arg = *Ts | *Tuple[...], ...]
        elif issubclass(type(arg), _Unpack_type) or issubclass(type(arg), types.GenericAlias):
            unpacked = getattr(arg, "__args__")[0] if issubclass(type(arg), _Unpack_type) else arg
            if isinstance(unpacked, TypeVarTuple):
                return [RepeatedDimensionArgInfo(UnboundAbstractDimensionArgInfo())]
            # unpacked is Tuple[T, ...] | Tuple[A, B, ...]
            if hasattr(unpacked, "__origin__") and getattr(unpacked, "__origin__") is tuple:
                # unpacked is Tuple[T, ...] i.e. T zero or more
                if len(getattr(unpacked, "__args__")) == 2 and getattr(unpacked, "__args__")[1] is ...:
                    # base = T
                    base = _unpack_recognize_arg(getattr(unpacked, "__args__")[0])[0]
                    if isinstance(
                        base,
                        ConcreteDimensionArgInfo | AbstractDimensionArgInfo | UnboundAbstractDimensionArgInfo,
                    ):
                        return [RepeatedDimensionArgInfo(base)]
                # unpacked is Tuple[A, B, ...] i.e. a bounded tuple of types
                r = []
                for a in getattr(unpacked, "__args__"):
                    r.extend(_unpack_recognize_arg(a))
                return r
        raise TypeError(f"Unrecognized shape parameter {arg} of type {type(arg)}")

    shape_args = []
    for arg in all_args[1:]:
        shape_args.extend(_unpack_recognize_arg(arg))

    shape_info = ShapeInfo(shape_args)
    if tensor is not None:
        if not shape_info.matches(tensor.size()):
            raise TypeError(f"Tensor size {tensor.size()} did not match shape arguments {shape_info}")

    return d_type, shape_info


def _is_repeated(
    arg: Optional[DimensionArgInfo],
) -> TypeGuard[RepeatedDimensionArgInfo]:
    return arg is not None and isinstance(arg, RepeatedDimensionArgInfo)
