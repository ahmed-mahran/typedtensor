from abc import ABCMeta
from typing import Type, cast


class DimensionMeta(ABCMeta):
    def __new__(cls, name, bases, namespace, **kwargs):
        if name != "Dimension":
            length = kwargs.get("length", None)
            if length is not None:
                namespace["length"] = length
        return super().__new__(cls, name, bases, namespace)


class Dimension(metaclass=DimensionMeta):
    length: int


def dimension(name: str, length: int) -> Type[Dimension]:
    return cast(Type[Dimension], DimensionMeta(name, (Dimension,), {}, length=length))


class Sub[T: Dimension](Dimension):
    pass


class Concat[A: Dimension, B: Dimension](Dimension):
    pass


class Rec[A: Dimension, T: Dimension](Dimension):
    """
    Recursive dimension is a type modifier that indicates that T is a recusrive type of A.
    T is a higher order type parameterized by A or recursively by T[] e.g. 
    T[A], T[T[A]], T[T[T[T[... T[A]]]]]]. All of those expansions are equivalent to Rec[A, T].

    Rec[A, T] is considered a sub-type of A

    ```
    Concat[Concat[ ... Concat[D, D] ... , D], D] = Rec[D, Concat]
    Concat[Past, Seq] = PastAndSeq
    PastAndSeq <= Past
    Concat[Past, Seq] <= Past
    Concat[Concat[Past, Seq], Seq] <= Past
    Rec[Past, Concat[_, Seq]] <= Past
    ```
    """

    pass
