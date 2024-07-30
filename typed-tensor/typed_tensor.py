from __future__ import annotations

import logging
from inspect import isclass
from typing import Callable, Optional, Tuple, Type, TypeGuard, cast, overload

import torch
from torch import Size, Tensor

from .dimension import Dimension, Z
from .shape_info import DimensionArgInfo, ShapeInfo, _extract_typed_args, _is_repeated
from .utils import _is_tensor_subclass, _is_type_var_of_bound

logger = logging.getLogger(__name__)


class TypedTensor[DType: Tensor, *Dimensions]:
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

    def _orig_class__setter(self, value):
        self._orig_class = value
        if self._args is None:
            self._args = value.__args__
            self._typed_args = None
            try:
                if TypedTensor.assert_valid_typed_tensor(self):
                    self._self_ = self
            except Exception as type_error:
                self._type_error = type_error

    __orig_class__ = property(
        fget=lambda self: self._orig_class, fset=_orig_class__setter
    )

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
            raise ValueError(
                f"Tensor sizes must match; provided {tensor.size()} while having {self.tensor.size()}"
            )
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

        def __getitem__[D0, D1](
            self, item: Tuple[D0, D1]
        ) -> TypedTensor[T, Z[Dimension], D0, D1]:
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
            return self.o.asinstanceof[
                TypedTensor[T, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]
            ]

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
        me = self.asinstanceof[
            TypedTensor[DType, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]]
        ]
        ts = list(me.args[1:])
        d0, d1 = ts[dim0], ts[dim1]
        ts[dim0] = d1
        ts[dim1] = d0
        return TypedTensor(
            cast(DType, me.tensor.transpose(dim0, dim1)), tuple([me.args[0]] + ts)
        )

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


def is_instance_of[DType: Tensor, *Dimensions, T](
    t: TypedTensor[DType, *Dimensions], tp: Type[T]
) -> TypeGuard[T]:
    tensor_dtype, tensor_shape_info = t.typed_args
    if hasattr(tp, "__args__"):
        type_dtype, type_shape_info = _extract_typed_args(getattr(tp, "__args__"))
    else:
        return False

    if isclass(type_dtype):
        if not _is_tensor_subclass(tensor_dtype, type_dtype):
            return False
    elif _is_type_var_of_bound(type_dtype, Tensor):
        if type_dtype.__bound__ is None or not _is_tensor_subclass(
            tensor_dtype, type_dtype.__bound__
        ):
            return False

    tensor_type_args = tensor_shape_info.args
    type_type_args = type_shape_info.args

    logger.debug(f"Is {tensor_type_args} <= {type_type_args}?")

    # Two sequences with non-repeating dimensions match if they have the same length and corresponding
    # entries at the same index match. Repeating dimensions adds complexity to the logic; we cannot
    # compare lengths of sequences to decide on matching. However, we can compare lengths of ordered
    # traversal steps. In that case, two sequences with possibly repeating dimensions match if there
    # are at least two complete and ordered traversals, one traversal per sequence, with same length
    # of steps, covering all items in the original sequence (hence complete) in the same order of items
    # in the original sequence (hence ordered), and with matching corresponding entries per step index.
    #
    # Traversal of a sequence of type arguments (e.g. [Dimension, BatchDim, *Dimensions, ...])
    # can be represented as a traversal on a directed graph, such that a node represents a state
    # of traversal and an edge represents a dimension that acts as a condition to determine the
    # next state of traversal.
    #
    # For the sequence [A, B, C, D], one state could be the initial state where we enter the
    # sequence, another state could be the final state where we terminate traversal at end
    # of the sequence, another state could be an intermediate state that we have reached
    # dimension B after A and we are ready to go to dimension C ...
    #
    # A non-repeating dimension has a single condition (or outgoing edge), it transitions the current
    # state to a single next state.
    # A repeating dimension has two conditions (or outgoing edges), it transitions the current state
    # back to itself or to a single next state.
    #
    # The non-minimized traversal graph is a chain of nodes with no loops involving more than one node.
    # This implies that we can traverse the graph from start to end passing by all nodes.
    #
    # If we could find any two matching traversals, then we could decide that both sequences match.
    #
    # This step function is a recursive function for searching two graphs for matching traversals.
    # At any time t (a step index in the traversal sequence), we keep an index i and an index j of
    # the current traversed nodes of both graphs. Each node i (or j) has possible next steps; if it
    # is a non-repeating node, then it has only one next step: the next node in the sequence (i + 1),
    # if it is a repeating node, then it has two possible next steps: the next node in the sequence
    # (i + 1) or back to itself (i). At each traversal step, we consider all possible next steps
    # , and we select the first one leading to a match.
    #
    # Repeated nodes could have zero or more matches. In case of zero match, the repeated node acts as
    # a skip connection between both nodes it is connecting (i.e. previous and next). In that case, a
    # repeated node adds one more skip edge from previous node to next node.
    #
    # If a traversal reaches the end of both graphs, then we conclude a match. Otherwise, a traversal
    # can be early terminated:
    # - if current nodes i and j are not matching
    # - either one of the graph has reached the end node while the other has not
    # - a special case is when a graph has reached the end node while the other
    #   has repeating nodes before the end node, in this case we consider the
    #   repeating nodes are matching for 0 occurrences in the other sequence
    #
    # [B, S] VS [d*, B, S]
    # _E0: epsilon_0, is the start condition
    #  _E: epsilon, is the termination condition
    #  _S: is the skip condition
    #
    #  _E0      B       S      _E
    # ----->()----->()----->()----->
    #       ..
    #       || d*
    #  _E0  v|  B       S      _E
    # -.--->()----.>()----->()----->
    #   \......../
    #       _S
    def step(
        i: int,
        j: int,
        indent: str,
        left: list[DimensionArgInfo],
        right: list[DimensionArgInfo],
    ):
        def line(content: str):
            logger.debug(f"{indent}{content}")

        t_i = tensor_type_args[i] if i < len(tensor_type_args) else None
        t_j = type_type_args[j] if j < len(type_type_args) else None
        if t_i is not None:
            left = [a for a in left] + [t_i]
        if t_j is not None:
            right = [a for a in right] + [t_j]
        line(f"step({left} <=> {right} || {i}: {t_i}, {j}: {t_j})")

        # both have terminated, that's a match
        if t_i is None and t_j is None:
            line("[ACCEPT] both terminated")
            return True

        # i has terminated while j has not
        if t_i is None:
            # if j is repeated (0 or more) just consume it (j + 1)
            # return step(i, j + 1) if _is_repeated(t_j) else False
            if _is_repeated(t_j):
                line("* i terminated but j is repeated")
                return step(i, j + 1, indent + "-", left, right)
            else:
                line("[REJECT] i terminated but j is not")
                return False
        # j has terminated while i has not
        if t_j is None:
            # if i is repeated (0 or more) just consume it (i + 1)
            # return step(i + 1, j) if _is_repeated(t_i) else False
            if _is_repeated(t_i):
                line("* j terminated but i is repeated")
                return step(i + 1, j, indent + "-", left, right)
            else:
                line("[REJECT] j terminated but i is not")
                return False
        # break on mismatch
        if not t_i.is_subclass(t_j):
            line("[REJECT] i is not subclass of j")
            return False

        # now consider all possible next steps
        # steps = []
        # if _is_repeated(t_i):
        #     steps += [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
        # else:
        #     steps += [(i + 1, j + 1)]
        #
        # if _is_repeated(t_j):
        #     steps += [(i, j + 1), (i + 1, j), (i + 1, j + 1)]
        # else:
        #     steps += [(i + 1, j + 1)]
        # steps = list(set(steps))
        # if the other node, t_j, is repeated, we add a skip edge from (j - 1) to (j + 1)
        # which is equivalent to moving from (i - 1, j - 1) to (i, j + 1)
        # which means that if we were at node (j - 1), one of the possible steps is to jump
        # to node (j + 1), or equivalently, if we are at node i and node j is repeated,
        # we can stay at node i and just move to next node (j + 1)
        t_i_steps = [i, i + 1] if _is_repeated(t_i) or _is_repeated(t_j) else [i + 1]
        t_j_steps = [j, j + 1] if _is_repeated(t_j) or _is_repeated(t_i) else [j + 1]
        # be careful not to stay at the same state where (i, j) = (i_step, j_step)
        steps = [
            (i_step, j_step)
            for i_step in t_i_steps
            for j_step in t_j_steps
            if i_step != i or j_step != j
        ]
        line(f"* steps: {steps}")
        for next_i, next_j in steps:
            # we have found a match in one of the possible next steps
            if step(next_i, next_j, indent + "-", left, right):
                return True
        # no matches found in any possible next step
        return False

    return step(0, 0, "", [], [])
