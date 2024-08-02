from typing import Type, cast


class DimensionMeta(type):
    def __new__(cls, name, bases, namespace, **kwargs):
        if name != "Dimension":
            length = kwargs.get("length", None)
            if length is not None:
                namespace["length"] = length
        return super().__new__(cls, name, bases, namespace)


class Dimension(metaclass=DimensionMeta):
    length: int


class Z[T: Dimension](Dimension):
    pass


def dimension(name: str, length: int) -> Type[Dimension]:
    return cast(Type[Dimension], DimensionMeta(name, (Dimension,), {}, length=length))


class Concat[A: Dimension, B: Dimension](Dimension):
    pass
