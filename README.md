# Typed Tensor

Yet another type annotations and runtime checking for tensor shape and datatype.

This is a python exercise trying to write statically typed and maybe functional style code by a scala minded person.

This is an opportunity to challenge python's static typing capabilities to express common patterns in neural networks. New python typing features shall be eagerly implemented by [MyPyright](https://github.com/ahmed-mahran/pyright).

Until all checks are static and have zero runtime overhead with advancement of [MyPyright](https://github.com/ahmed-mahran/pyright) or python, this can mostly serve pedagogical puposes teaching and learning neural networks.

---
**<p style="text-align: center;">\*\*\*\* NOTE \*\*\*\*</p>**
This goes hand in hand with [MyPyright](https://github.com/ahmed-mahran/pyright) as it depends on features **NOT** yet supported by [python typing PEPs](https://peps.python.org/topic/typing/).

---

---
**<p style="text-align: center;">\*\*\*\* NOTE \*\*\*\*</p>**
This and [MyPyright](https://github.com/ahmed-mahran/pyright) are work under development.

---

# Example

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
from typedtensor import Dimension

...

# Define each dimension as a class.
# Yes, we should be that explicit.
class BatchDim(Dimension, length=128): pass
class SequenceDim(Dimension): pass
class FeatureDim(Dimension): pass

# matmul method signature defined in TypedTensor
# *Ds is a variadic generic which implies zero or
# more Dimensions
def matmul[*Ds, D0, D1, D2](
    self: TypedTensor[DType, *Ds, D0, D1],
    other: TypedTensor[DType, *Ds, D1, D2],
) -> TypedTensor[DType, *Ds, D0, D2]:
    ...

# this approximates transpose method signature in TypedTensor
def transpose[*Init, D0, *Mid, D1, *Tail](
    self: TypedTensor[DType, *Init, D0, *Mid, D1, *Tail],
) -> TypedTensor[DType, *Init, D1, *Mid, D0, *Tail]:
    ...

a = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))
b = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))


# TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, SequenceDim]
w = (
    a # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
    .matmul(
        b # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
        .transpose[SequenceDim, FeatureDim] # TypedTensor[FloatTensor, BatchDim, FeatureDim, SequenceDim]
    )  # TypedTensor[FloatTensor, BatchDim, SequenceDim, SequenceDim]
)
```

Even more, see [typed GPT2](examples/ttransformers/models/gpt2/modeling_gpt2.py) ...
# Install

---
**<p style="text-align: center;">\*\*\*\* NOTE \*\*\*\*</p>**
This package is best compatible with [`MyPyright`](https://marketplace.visualstudio.com/items?itemName=mashin.mypyright) and `VSCode`. Otherwise, other static type checkers will generate errors as `typedtensors` depends on features NOT yet supported by [python typing PEPs](https://peps.python.org/topic/typing/).

---

`typedtensor` is not yet published however if you want to give it a try locally:

```commandline
pip install -U git+https://github.com/ahmed-mahran/typedtensor.git@main
```

To remove it:

```commandline
pip uninstall typedtensor
```

To view `examples`, you may clone this repo. Again, You must use `MyPyright` as a type checker.

```commandline
git clone git@github.com:ahmed-mahran/typedtensor.git
# Or
git clone https://github.com/ahmed-mahran/typedtensor.git
```

# Status quo

This has been quite a challenging endeavour for mainly two reasons:
- Python is not built up to be a statically typed language.
- Tensor operations have inherently complex patterns and relations that could hardly be captured by any ordinary type system.

Static typing features were incrementally added to python through [a series of PEP's](https://peps.python.org/topic/typing/). In particular, [PEP 646](https://peps.python.org/pep-0646/) was a major milestone which has introduced variadic generics which allows the type of array-like structures to be parameterised with the array shape. On the other hand tensor operations have complex type patterns. Transpose and shape permute operations would require rearranging shape type parameters. Concatenation and stacking operations would require some sort of type parameters aggregations to type-hint affected dimensions. Not to mention convolution operations which would require some sort of type arithmetics to express output dimensions as a function of dimensions of input tensors and input parameters like stride, padding and dilation.

Python current typing specifications are not sufficient to seamlessly express all tensor operations. Moreover, it is rather slow to implement new specifications; it could take years to discuss and implement a single PEP and to see that PEP effective in all type checkers. On the other hand tensor ops libraries are being developed in fast pace and are being adopted further by a fast paced well adpoted libraries, e.g. pytorch -> transformers. All those libraries are written in a pythonic way which would require re-writing and re-structuring to adhere to static type safety.

# Details

## Basic types
### Dimension
A tensor shape is described by a sequence of dimension types. `Dimension` is the base class for all dimension types. For each dimension type, there should be a class extending directly or indirectly from `Dimension`. Dimension size can be captured by setting class variable `length`.

For example:

```python
# Abstract Batch dimension with no length
class BatchDim(Dimension): pass
# Concrete Feature dimension with a length
class FeatureDim(Dimension, length=768): pass
# Abstract sequence dimensions related by certain type hierarchy
class _SequenceDim(Dimension): pass
class CurrentSequenceDim(_SequenceDim): pass
class PastAndCurrentSequenceDim(_SequenceDim): pass
```

`typedtensor` discourages usage of literals as types, strings or numbers, to describe shape. User defined types (UDT's) are more strict, type-safe and IDE friendly. UDT's can capture structural semantic relations through type hierarchy. The ability to organize types in a hierarchy is essential to determine types equivalence relations which is essential to typing tensor operations demanding less complex features from the type system.

### TypedTensor [`DType`, `*Ds`]

`TypedTensor` is the typed wrapper for any `torch.Tensor`. It is parameterized by:

- `DType`: which specifies the wrapped tensor type, e.g. `torch.Tensor` or `torch.FloatTensor`
- `*Ds`: which is a variadic type variable of dimension types describing the order and types of tensor shape

For example:

```python
TypedTensor[torch.FloatTensor, BatchDim, CurrentSequenceDim, FeatureDim]
TypedTensor[torch.LongTensor, BatchDim, CurrentSequenceDim]
```

Ideally shape dimensions should be unique, otherwise this can cause ambiguity matching shapes and accessing/referencing/manipulating dimensions. Currently, `typedtensor` doesn't impose a uniqueness constrain on types of shape dimensions however this may be added in future.

### Shape [`*Ds`]

Holds specific dimension types. Useful to pass specific dimension types around as type arguments. Can be used to get around limitations of variadic type variables. E.g. functions parameterized by variadic type variables would capture the type of individual type parameters. For example, calling `fn[*Ds](*types: *Ds) -> TypedTensor[Tensor, *Ds]: ...` as `fn(Batch, Seq)` would return `TypedTensor[Tensor, Type[Batch], Type[Seq]]` however we would expect to be able to return `TypedTensor[Tensor, Batch, Seq]` instead. `typedtensor` uses `Shape` to get around this limitation.

In our example, `fn` would be defined as `fn[*Ds](types: ShapeArgs[*Ds]) -> TypedTensor[Tensor, *Ds]: ...` and called as `fn(Shape[Batch, Seq])` to retrun `TypedTensor[Tensor, Batch, Seq]`.

Also, `Shape` is treated as a special dimension that is equivalent to unpacked tuple `*Tuple[*Ds]`. Moreover, it can be nested in shape definition, e.g. `TypedTensor[Tensor, Batch, Seq, Head, Feature]`, `TypedTensor[Tensor, Shape[Batch, Seq, Head, Feature]]`, `TypedTensor[Tensor, Shape[Shape[Batch, Seq], Shape[Head], Feature]]` ... are all equivalent.

## Broadcasting

### Semantics

Broadcast semantics (see [pytorch-broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html)) on type level are defined differently. Shapes are aligned from right to left. Dimension types from the higher dimensional shape on the left that don't align with any dimension type from the lower dimensional shape are retruned as-is. For each pair of the aligned dimension types, the type with longer length is returned given that the other has length 1, or otherwise the type which is super to the other type is returned, or otherwise broadcasting fails.

Currently broadcasting doesn't handle repeated dimensions. This is because repeated dimensions shouldn't be used at runtime. However, if this logic to be run at static type checking time, repeated dimensions must be handled somehow.

### Broadcast [`Shape`, `Shape`]

This is the type of a broadcasted shape, which is by itself a special dimension like `Shape`. Hence, it can be recursively nested to express broadcasted multiple shapes.

```python
def where[DType: Tensor, *Cs, *Is, *Os](
    condition: TypedTensor[BoolTensor, *Cs],
    input: TypedTensor[DType, *Is],
    other: TypedTensor[DType, *Os]
) -> TypedTensor[DType, Broadcast[Shape[Broadcast[Shape[*Cs], Shape[*Is]]], Shape[*Os]]]:
    ...
```

`Broadcast` is a binary operation on shapes: `Shape x Shape -> Shape`. The static type system needs to evaluate that operation to decide on type equivalence and to infer shape types.

### Sub [`D`]

`Sub` convinces the type system that `Sub[D]` is a sub-type of `D`. This is for convinience to avoid defining redundant sub-types. Sometimes it will be cumbersome or not straightforward to define the dimension type of a broadcastable tensor. For example, when applying a mask on a tensor `x` using  `where(mask_bool, x, mask_value)`, if `mask_value` has dimensions different than `x` either in length per dimension or number of dimensions and we don't bother about dimension types of `mask_value`, shape of `mask_value` can be defined with sub-types of dimensions of `x`. `Sub[D]` can be used here to define shape of `mask_value` and hence broadcasting can return `D`.

```python
x: TypedTensor[Tensor, Batch, Head, Seq1, Seq2] = ...
mask_value: TypedTensor[Tensor, Sub[Seq2]] = ... # if we cannot use Seq2 here
mask_bool: TypedTensor[BoolTensor, Sub[Seq1], Sub[Seq2]] = ... # if we cannot use Seq1 or Seq2 here
masked_x = where(mask_bool, x, mask_value)
reveal_type(masked_x) # TypedTensor[Tensor, Broadcast[Shape[Broadcast[Shape[Sub[Seq1], Sub[Seq2]], Shape[Batch, Head, Seq1, Seq2]]], Shape[Sub[Seq2]]]]
masked_x.asinstanceof[TypedTensor[Tensor, Batch, Head, Seq1, Seq2]] # ACCEPTABLE
```

<!---
### `Concat[D0, D1]`
### `Rec[D, F]`
-->

# New typing requirements

In this section, we highlight typing features that are missing in python however are required for tensors type system. This section will be expanded when new features are added. Also, these features are/will be supported by [MyPyright](https://github.com/ahmed-mahran/pyright).

## Multiple Variadic Generics

[![MyPyright Support](https://img.shields.io/badge/MyPyright-Yes-green)](https://github.com/ahmed-mahran/mypyright#multiple-unpackings-of-type-arguments)


Multiple zero or more dimensions! PEP 646 [doesn't allow multilpe variadic type variables](https://peps.python.org/pep-0646/#multiple-type-variable-tuples-not-allowed) nor it allows [multiple unpackings](https://peps.python.org/pep-0646/#multiple-unpackings-in-a-tuple-not-allowed). However, arbitrarly dimension picking operations, like transpose or concatenate, need to describe shape patterns with more than one wildcard. For example, consider the transpose operation:

```python
def transpose[*PreDs, *MidDs, *PostDs, D0, D1](
    self: TypedTensor[DType, *PreDs, D0, *MidDs, D1, *PostDs],
) -> TypedTensor[DType, *PreDs, D1, *MidDs, D0, *PostDs]:
    ...
```

Transpose swaps any two dimensions `D0` and `D1`. The input tensor should be of the form `TypedTensor[DType, *PreDs, D0, *MidDs, D1, *PostDs]` while the output tensor should be of the same form but with `D0` and `D1` swapped `TypedTensor[DType, *PreDs, D1, *MidDs, D0, *PostDs]`. It is currently not possibel to write such type patterns in python. `typedtensor` tried to implement a new generic type, `Z[D]`, which corresponds to zero or more dimensions to fulfill this need

```python
def transpose[D0, D1](
    self: TypedTensor[DType, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]],
) -> TypedTensor[DType, Z[Dimension], D1, Z[Dimension], D0, Z[Dimension]]:
    ...
```

This comes with an extra cost and redundancy as we need explicit type casting to convince the type checker that a typed tensor is of a certain shape pattern. E.g. in order to use transposed tensor, it should first be cast to the suitable shape pattern.

```python
a: TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim] = ...

a_t: TypedTensor[torch.FloatTensor, BatchDim, FeatureDim, SequenceDim] = (
    a # TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]
    .transpose[SequenceDim, FeatureDim] # TypedTensor[FloatTensor, Z[Dimension], FeatureDim, Z[Dimension], SequenceDim]
    .asinstanceof[TypedTensor[FloatTensor, BatchDim, FeatureDim, SequenceDim]]
)
```

## Type Transformations of Variadic Generics

[![MyPyright Support](https://img.shields.io/badge/MyPyright-Yes-green)](https://github.com/ahmed-mahran/mypyright#type-transformations-of-variadic-type-variables)

Passing dimensions classes as arguments through variadic type variables as parameters would capture the type of the dimension class and not the class itself, e.g. `Type[Batch]` instead of `Batch`. This makes it impossible to describe some tensor operations as they would produce tensors like `TypedTensor[Tensor, Type[Batch], Type[Features]]` instead of `TypedTensor[Tensor, Batch, Features]`.

```python
fn[*Ds](*types: *Ds) -> TypedTensor[Tensor, *Ds]: ...
reveal_type(fn(Batch, Feature)) # TypedTensor[Tensor, Type[Batch], Type[Features]]
```

`typedtensor` tried a workaround by defining a new shape container, `Shape[*Ds]` and type alias `ShapeArgs[*Ds] = Type[Shape[*Ds]]`.

```python
fn[*Ds](types: Type[Shape[*Ds]]) -> TypedTensor[Tensor, *Ds]: ...
reveal_type(fn(Shape(Batch, Feature))) # TypedTensor[Tensor, Batch, Features]
```

It will be more convenient if python allows type transformations on variadic generics.

```python
fn[*Ds](types: *Map[Type, *Ds]) -> TypedTensor[Tensor, *Ds]: ...
reveal_type(fn(Batch, Feature)) # TypedTensor[Tensor, Batch, Features]
```

## Subscriptable Functions

[![MyPyright Support](https://img.shields.io/badge/MyPyright-Yes-green)](https://github.com/ahmed-mahran/mypyright#subscriptable-functions)

It is common in typed tensor operations to pass dimension types as function arguments. This is a major goal of typed tensor operations anyways; we don't want to pass around dimension indices anymore. Subscriptable functions imposes itself mainly for the following reasons:

- As a syntactic sugar: Type arguments are more naturally fit where type parameters are defined/expected. If you define a generic function `def fn[T](tp: Type[T])`, it feels more natural to pass a type argument such as `int` between the square brackets, i.e. `fn[int]()` instead of `fn(int)`.
- As a functional need: It is crucial to be able to bind certain type parameters to specific type arguments regardless from the order of function parameters. This is crucial especially in methods where `self` argument needs to be annotated using certain type parameters that appear next in method signature. Also, when multiple variadic type variables are used in `self` annotation, pre- and post-binding of a type variable can lead to different variadic type variables assignment.

Consider the following `transpose` operation which is supposed to swap two dimensions:

```python
class Tensor[*Shape]:
  def transpose[*Init, D1, *Mid, D2, *Tail](
    self: Tensor[*Init, D1, *Mid, D2, *Tail],
    d1: Type[D1],
    d2: Type[D2]
  ) -> Tensor[*Init, D2, *Mid, D1, *Tail]: ...
```

The `transpose` operation matches any tensor shape with the pattern `[*Init, D1, *Mid, D2, *Tail]`. It is only interested in the two dimensions `D1` and `D2` that will swap places in the same shape pattern to result in the shape `[*Init, D2, *Mid, D1, *Tail]`.

```python
x = Tensor[A, B, C, D]()
reveal_type(x.transpose(B, D)) # Tensor[A, B, D, C]
# Error:
# Argument of type "type[B]" cannot be assigned to parameter "d1" of type "type[C]" in function "transpose"
#  "type[B]" is not assignable to "type[C]"
```

The type checker has matched the type parameters of `self: Tensor[*Init, D1, *Mid, D2, *Tail]` first leading to the assignment: `*Init = [A, B], D1 = C, *Mid = [], D2 = D, *Tail = []`. Then by substituting these assignements in the rest of the function parameters and return type, the method signature becomes:

```python
# Note: this is an invalid syntax for purpose of illustration.
# We could use *tuple though, but no need to add more unneccessary complexity.
def transpose(
  self: Tensor[*[A, B], C, *[], D, *[]],
  d1: Type[C],
  d2: Type[D]
) -> Tensor[*[A, B], D, *[], C, *[]]: ...
```

This explains the return type of `Tensor[A, B, D, C]` instead of the correct and expected `Tensor[A, D, C, B]`. This also explains type error in function first argument, the method expects `d1` of type `Type[C]` while the caller has passed an argument of type `Type[B]`.

This should be resolved with subscriptable functions.

```python
x = Tensor[A, B, C, D]()
reveal_type(x.transpose[B, D]()) # Tensor[A, D, C, B]
```

The type checker should match the type parameters of `transpose` first leading to the assignment `D1 = B, D2 = D`. Then by substituting these assignements in the rest of the function parameters and return type, the method signature becomes:

```python
def transpose(
  self: Tensor[*Init, B, *Mid, D, *Tail],
  d1: Type[B],
  d2: Type[D]
) -> Tensor[*Init, D, *Mid, B, *Tail]: ...
```

Now the type checker can correctly and as intended match function parameters with arguments and infer return type. Matching arguments to parameters leads to the assignment: `*Init = [A], *Mid = [C], *Tail = []` and hence a correct return type `Tensor[A, D, C, B]`.
