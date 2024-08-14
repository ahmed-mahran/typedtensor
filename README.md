# Typed Tensor
Yet another type annotations and runtime checking for tensor shape and datatype.

This is a python exercise trying to write statically typed and maybe functional style code by a scala minded person.

This is an opportunity to challenge python's static typing capabilities to express common patterns in neural networks. 

This can mostly serve pedagogical puposes teaching and learning neural networks.

## Example

Before ...
```python
a = torch.randn(128, 1024, 768)
```

... After
```python
a = TypedTensor[torch.Tensor, BatchDim, SequenceDim, FeatureDim](torch.randn(128, 1024, 768))
```

A little bit more ...

```python
from torch import (Tensor, FloatTensor)
from typedtensor import (
  Dimension,
  Z,
)

...

# Define each dimension as a class.
# Yes, we should be that explicit.
class BatchDim(Dimension, length=128): pass
class SequenceDim(Dimension): pass
class FeatureDim(Dimension): pass

# matmul method signature defined in TypedTensor
# Z[Dimension] corresponds to a variadic generic which implies zero or
# more Dimensions
def matmul[D0, D1, D2](
    self: TypedTensor[DType, Z[Dimension], D0, D1],
    other: TypedTensor[DType, Z[Dimension], D1, D2],
) -> TypedTensor[DType, Z[Dimension], D0, D2]:
    ...

# this approximates transpose method signature in TypedTensor
def transpose[D0, D1](
    self: TypedTensor[DType, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]],
) -> TypedTensor[DType, Z[Dimension], D1, Z[Dimension], D0, Z[Dimension]]:
    ...

a = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))
b = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))


# NOTE: type casting ops, with name as*, can be removed if new specs are imlemented in python and type checkers
# TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, SequenceDim]
w = (
    a # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
    .as_z_d0_d1[SequenceDim, FeatureDim] # TypedTensor[FloatTensor, Z[Dimension], SequenceDim, FeatureDim]
    .matmul(
        b # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
        .transpose[SequenceDim, FeatureDim] # TypedTensor[FloatTensor, Z[Dimension], FeatureDim, Z[Dimension], SequenceDim]
        .as_z_d0_d1[FeatureDim, SequenceDim] # TypedTensor[FloatTensor, Z[Dimension], FeatureDim, SequenceDim]
    )  # TypedTensor[FloatTensor, Z[Dimension], SequenceDim, SequenceDim]
).asinstanceof[TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, SequenceDim]]
```

Ideally ...

```python
class BatchDim(Dimension, length=128): pass
class SequenceDim(Dimension): pass
class FeatureDim(Dimension): pass

def matmul[*Ds, D0, D1, D2](
    self: TypedTensor[DType, *Ds, D0, D1],
    other: TypedTensor[DType, *Ds, D1, D2],
) -> TypedTensor[DType, *Ds, D0, D2]:
    ...

def transpose[*PreDs, *MidDs, *PostDs, D0, D1](
    self: TypedTensor[DType, *PreDs, D0, *MidDs, D1, *PostDs],
) -> TypedTensor[DType, *PreDs, D1, *MidDs, D0, *PostDs]:
    ...

a = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))
b = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))


# TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, SequenceDim]
w = (
    a # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
    .matmul(
        b # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
        .transpose[SequenceDim, FeatureDim] # TypedTensor[FloatTensor, BatchDim, FeatureDim, SequenceDim]
    )  # TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, SequenceDim]
)
```

Even more, see [typed GPT2](examples/ttransformers/models/gpt2/modeling_gpt2.py) ...

## Status quo

This has been quite a challenging endeavour for mainly two reasons:
- Python is not built up to be a statically typed language.
- Tensor operations have inherently complex patterns and relations that could hardly be captured by any ordinary type system.

Static typing features were incrementally added to python through [a series of PEP's](https://peps.python.org/topic/typing/).
In particular, [PEP 646](https://peps.python.org/pep-0646/) was a major milestone which has introduced variadic generics which allows
the type of array-like structures to be parameterised with the array shape. On the other hand tensor operations
have complex type patterns. Transpose and shape permute operations would require rearranging shape type parameters.
Concatenation and stacking operations would require some sort of type parameters aggregations to type-hint affected
dimensions. Not to mention convolution operations which would require some sort of type arithmetics to express
output dimensions as a function of dimensions of input tensors and input parameters like stride, padding and dilation.

Python current typing specifications are not sufficient to seamlessly express all tensor operations. Moreover,
it is rather slow to implement new specifications; it could take years to discuss and implement a single PEP
and to see that PEP effective in all type checkers. On the other hand tensor ops libraries are being developed
in fast pace and are being adopted further by a fast paced well adpoted libraries, e.g. pytorch -> transformers.
All those libraries are written in a pythonic way which would require re-writing and re-structuring to adhere
to static type safety.