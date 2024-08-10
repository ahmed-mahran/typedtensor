from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, List, Optional, Tuple, Type, TypeGuard, TypeVar, Unpack, cast

import torch

# typing._UnpackGenericAlias
_Unpack_type = type(Unpack[...])  # type: ignore
# typing._GenericAlias
_GenericAlias = type(Tuple[Any])


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
            arg_bound is not None and bound is not None and _is_tensor_subclass(arg_bound, bound)
        )
    return False


def _is_generic_type[T](arg, tp: Type[T]) -> bool:
    return (
        issubclass(type(arg), _GenericAlias)
        and hasattr(arg, "__origin__")
        and getattr(arg, "__origin__") is tp
        and hasattr(arg, "__args__")
    )


class CaptureTypeArgs(ABC):
    def _orig_class__setter(self, value):
        self._orig_class = value
        self._on_type_args(value.__args__)

    __orig_class__ = property(fget=lambda self: self._orig_class, fset=_orig_class__setter)

    @abstractmethod
    def _on_type_args(self, type_args: Tuple[Type, ...]):
        pass


class CapturedTypeArgs(CaptureTypeArgs):
    def _on_type_args(self, type_args: Tuple[Type, ...]):
        self._type_args = type_args

    @property
    def type_args(self):
        return self._type_args


# def function_capture_type_args_call[R, **P](fn: Callable[Concatenate[Tuple[Type, ...], P], R]):
#     class _CaptureTypeArgs:
#         def __getitem__(self, item):
#             self.item = item
#             return self

#         def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
#             return fn(self.item, *args, **kwargs)

#     return _CaptureTypeArgs()


# def function_capture_type_args_getitem[R](fn: Callable[[Tuple[Type, ...]], R]):
#     class _CaptureTypeArgs:
#         def __getitem__(self, item):
#             return fn(item)

#     return _CaptureTypeArgs()


# def method_capture_type_args_call[R, **P](fn: Callable[Concatenate[Any, Tuple[Type, ...], P], R]):
#     class _CaptureTypeArgs:
#         def __get__(self, instance, owner):
#             self.instance = instance
#             return self

#         def __getitem__(self, item):
#             self.item = item
#             return self

#         def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
#             return fn(self.instance, self.item, *args, **kwargs)

#     return _CaptureTypeArgs()


# def method_capture_type_args_getitem[I, **P, R](fn: Callable[Concatenate[I, P], R]):
#     class _CaptureTypeArgs[II]:
#         def __get__(self, instance: II, owner: Type[II]):
#             self.instance = instance
#             return self

#         def __getitem__(self, *item: P.args, **kwargs: P.kwargs):
#             args = item[0] if type(item[0]) is tuple else item
#             return fn(cast(I, self.instance), *args, **kwargs)

#     return _CaptureTypeArgs[I]()


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
def match_sequence[A, B](
    a_sequence: List[A],
    b_sequence: List[B],
    is_repeated_a: Callable[[Optional[A]], bool],
    is_repeated_b: Callable[[Optional[B]], bool],
    a_matches_b: Callable[[A, B], bool],
    logger: Logger,
):
    def step(i: int, j: int, indent: str, left: List[A], right: List[B]):
        def line(content: str):
            logger.debug(f"{indent}{content}")

        t_i = a_sequence[i] if i < len(a_sequence) else None
        t_j = b_sequence[j] if j < len(b_sequence) else None
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
            # return step(i, j + 1) if is_repeated_b(t_j) else False
            if is_repeated_b(t_j):
                line("* i terminated but j is repeated")
                return step(i, j + 1, indent + "-", left, right)
            else:
                line("[REJECT] i terminated but j is not")
                return False
        # j has terminated while i has not
        if t_j is None:
            # if i is repeated (0 or more) just consume it (i + 1)
            # return step(i + 1, j) if is_repeated_a(t_i) else False
            if is_repeated_a(t_i):
                line("* j terminated but i is repeated")
                return step(i + 1, j, indent + "-", left, right)
            else:
                line("[REJECT] j terminated but i is not")
                return False
        # break on mismatch
        if not a_matches_b(t_i, t_j):
            line("[REJECT] i is not subclass of j")
            return False

        # now consider all possible next steps
        # steps = []
        # if is_repeated_a(t_i):
        #     steps += [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
        # else:
        #     steps += [(i + 1, j + 1)]
        #
        # if is_repeated_b(t_j):
        #     steps += [(i, j + 1), (i + 1, j), (i + 1, j + 1)]
        # else:
        #     steps += [(i + 1, j + 1)]
        # steps = list(set(steps))
        # if the other node, t_j, is repeated, we add a skip edge from (j - 1) to (j + 1)
        # which is equivalent to moving from (i - 1, j - 1) to (i, j + 1)
        # which means that if we were at node (j - 1), one of the possible steps is to jump
        # to node (j + 1), or equivalently, if we are at node i and node j is repeated,
        # we can stay at node i and just move to next node (j + 1)
        t_i_steps = [i, i + 1] if is_repeated_a(t_i) or is_repeated_b(t_j) else [i + 1]
        t_j_steps = [j, j + 1] if is_repeated_b(t_j) or is_repeated_a(t_i) else [j + 1]
        # be careful not to stay at the same state where (i, j) = (i_step, j_step)
        steps = [(i_step, j_step) for i_step in t_i_steps for j_step in t_j_steps if i_step != i or j_step != j]
        line(f"* steps: {steps}")
        for next_i, next_j in steps:
            # we have found a match in one of the possible next steps
            if step(next_i, next_j, indent + "-", left, right):
                return True
        # no matches found in any possible next step
        return False

    return step(0, 0, "", [], [])
