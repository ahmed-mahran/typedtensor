from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type, TypeGuard, TypeVar, cast

import torch


def _is_tensor_subclass[T, P](tp: Type[T], parent: Type[P]) -> bool:
    if f"{parent.__module__}.{parent.__name__}" in [
        "torch.Tensor",
        "torch.TensorBase",
    ] and tp in [
        torch.DoubleTensor,
        torch.FloatTensor,
        torch.BFloat16Tensor,
        torch.LongTensor,
        torch.IntTensor,
        torch.ShortTensor,
        torch.HalfTensor,
        torch.CharTensor,
        torch.ByteTensor,
        torch.BoolTensor,
    ]:
        return True
    return issubclass(tp, parent)


def _is_type_var_of_bound(arg: Any, bound: Optional[Type] = None) -> TypeGuard[TypeVar]:
    if issubclass(type(arg), TypeVar):
        arg_bound = cast(TypeVar, arg).__bound__
        return (arg_bound is None and bound is None) or (
            arg_bound is not None
            and bound is not None
            and _is_tensor_subclass(arg_bound, bound)
        )
    return False

class CaptureTypeArgs(ABC):
    def _orig_class__setter(self, value):
        self._orig_class = value
        self._on_type_args(value.__args__)

    __orig_class__ = property(
        fget=lambda self: self._orig_class, fset=_orig_class__setter
    )

    @abstractmethod
    def _on_type_args(self, type_args: Tuple[Type, ...]):
        pass
