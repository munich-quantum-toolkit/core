# Call-aware linear-value tracking and builder ownership

Status: superseded in PR #2336 by the
[QCO function model](qco-function-model.md). Historical record for the work
associated with PRs #2194 and #2196.

## Outcome and decisions

The earlier QCO and QTensor call mapping derived argument/result correspondence
from the callee body. It returned `FailureOr<Value>` for unsupported
declarations, recursion, incomplete bodies, and multi-block bodies. Positional
pairing is insufficient without an enforced ABI because it can join unrelated
linear values.

Module-wide static-root validation did not establish call correspondence.
Mapping consumers used type-filtered operands belonging to the call, and callee
mutation required cache invalidation. The successor removed this inference and
its caches: generic calls now end local traversal, while explicit unitary calls
and the validated positional function ABI supply the needed correspondence.

The earlier builder used these mappings to transfer tracked values across calls.
The requirement to complete helpers before adding entry-point operations
survives in the successor's complete callback APIs. This avoids partially built
tracking state and keeps leak detection and outer-value ownership explicit.

Historical tests covered reordered correspondence, retained and created values,
unsupported declarations and recursion, and builder ownership. Their results
describe the earlier contract; consult the successor for current acceptance
criteria and test entry points.

## Source and validation

- `mlir/include/mlir/Dialect/QCO/Utils/WireIterator.h` and its implementation
  own scalar-qubit traversal.
- `mlir/include/mlir/Dialect/QTensor/Utils/TensorIterator.h` and its
  implementation own tensor traversal.
- `mlir/include/mlir/Dialect/QCO/Builder/QCOProgramBuilder.h` and its
  implementation own function construction and tracked values.

Historical validation passed the focused QCO utility, QTensor utility, builder,
and QCO IR suites, the release build, configured CTests with one expected skip,
and both lint sessions. Hosted CI was not monitored. Those results are not a
fresh validation of the current checkout.
