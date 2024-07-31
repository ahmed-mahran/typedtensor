# typedtensor
Yet another type annotations and runtime checking for tensor shape and datatype.

This is a python excersie trying to write statically typed and maybe functional style code by a scala minded person.

This is an opportunity to challenge python's static typing capabilities to express common patterns in neural networks. 

This can mostly serve pedagogical puposes teaching and learning neural networks.

## Example
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

# transpose method signature defined in TypedTensor
# Z[Dimension] corresponds to a variadic generic which implies zero or
# more Dimensions
def transpose[D0, D1](
      self: TypedTensor[DType, Z[Dimension], D0, Z[Dimension], D1, Z[Dimension]],
      dim0: int,
      dim1: int,
  ) -> TypedTensor[DType, Z[Dimension], D1, Z[Dimension], D0, Z[Dimension]]:
  ...

x_0 = TypedTensor[torch.FloatTensor, BatchDim, SequenceDim, FeatureDim](cast(torch.FloatTensor, torch.randn(128, 1024, 768)))
reveal_type(x_0) # Type of "x_0" is "TypedTensor[FloatTensor, BatchDim, SequenceDim, FeatureDim]"
x_1 = x_0.as_z_d0_z_d1_z[BatchDim, FeatureDim]
reveal_type(x_1) # Type of "x_1" is "TypedTensor[FloatTensor, Z[Dimension], type[BatchDim], Z[Dimension], type[FeatureDim], Z[Dimension]]"
x_2 = x_1.transpose(0, 2)
reveal_type(x_2) # Type of "x_2" is "TypedTensor[FloatTensor, Z[Dimension], type[FeatureDim], Z[Dimension], type[BatchDim], Z[Dimension]]"
x_3 = x_2.asinstanceof[TypedTensor[torch.FloatTensor, FeatureDim, SequenceDim, BatchDim]]
reveal_type(x_3) # Type of "x_3" is "TypedTensor[FloatTensor, FeatureDim, SequenceDim, BatchDim]"

```