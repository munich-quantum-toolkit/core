# Value-semantic functions and calls in QCO

Status: complete in PR #2336 (`f57b6c341`). Historical implementation record;
extends the [QC function model](qc-function-model.md) and supersedes
[body-derived generic-call mapping](rebase-call-mapping-stack.md).

## Outcome and scope

QCO supports reusable unitary definitions through `qco.call` and uses a
positional ABI for ordinary scalar-qubit functions. QC/QCO conversions preserve
functions and calls within the supported one-block outer function shape. Generic
calls stop local wire traversal; unitary calls expose direct input/output
correspondence without inlining or matrix synthesis.

## Decisions

- A QCO function returns source-language results first, followed by one updated
  qubit per scalar qubit argument in argument order. The target formats borrow
  fixed operands; an explicit positional convention avoids result annotations
  that can become stale independently of a signature. QCO-to-QC validates
  correspondence before stripping these trailing results.
- A marked QCO unitary function accepts `f64` parameters followed by qubits and
  returns those qubits positionally. The marker verifier traces returned qubits
  through unitary operations to their corresponding arguments. `qco.call`
  exposes this mapping through `UnitaryOpInterface`; its compile-time matrix is
  unknown.
- Generic `func.call` ends local wire and tensor traversal. Such functions may
  measure, reset, allocate, branch, or return unrelated values. Interprocedural
  consumers use the declared ABI and validate the supported correspondence;
  local iterators no longer infer whole-callee behavior or maintain
  `CallQubitMapping` and `CallTensorMapping` caches.
- The builder takes complete callbacks and same-module function handles, infers
  results, validates positional qubit returns, and updates live values at calls.
  This keeps insertion and linear tracking state local to function construction
  instead of exposing paired start/end APIs.
- QC-to-QCO appends the latest value of each borrowed qubit argument to the
  return. State maps retain the original QC SSA argument as their key; using the
  converted QCO value as a second key would return a stale value. Duplicate or
  explicitly returned borrowed qubits are rejected before conversion mutates IR.
- QCO-to-QC strips validated trailing qubit results from signatures, returns,
  and call sites, substituting the matching operands. Earlier results remain
  ordinary converted results. Both passes temporarily remove the unitary marker
  while signature conversion creates intermediate casts, then restore it on
  success or failure.
- Call argument, result, and discardable attributes must survive conversion.
  Nonempty metadata on synthetic QCO pass-through results is rejected when QC
  has no place to represent it. Silent metadata loss is not a supported
  conversion.
- Call correspondence accessors and the enclosing function verifier must be safe
  on malformed IR: attribute verification can query the unitary interface before
  the call's own verifier runs. Safety cannot depend on verifier order.

## Source and validation

- `mlir/lib/Dialect/MQT/IR/MQTDialect.cpp` and
  `mlir/lib/Dialect/QCO/IR/Operations/CallOp.cpp` own unitary function and call
  verification.
- `mlir/lib/Dialect/QCO/Utils/FunctionUtils.cpp` owns positional argument
  tracing; `WireIterator.cpp` in that directory and
  `mlir/lib/Dialect/QTensor/Utils/TensorIterator.cpp` own local traversal.
- `mlir/lib/Dialect/QCO/Builder/QCOProgramBuilder.cpp` owns helper creation and
  live-value tracking.
- `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` and
  `mlir/lib/Conversion/QCOToQC/QCOToQC.cpp` own ABI and metadata conversion.

The QCO IR, utility, conversion, and `QCQCORoundTrip` tests cover unitary
helpers, modifiers, generic-call boundaries, malformed calls, preserved or
rejected metadata, and reconstruction of the positional ABI. Focused checks from
the repository root include:

```sh
cmake --build --preset release --target mqt-core-mlir-unittest-qco-ir mqt-core-mlir-unittest-qco-utils mqt-core-mlir-unittest-qc-to-qco mqt-core-mlir-unittest-qco-to-qc
ctest --test-dir build/release -R 'QCO|QCToQCO|QCOToQC' --output-on-failure
```

The [QC companion](qc-function-model.md#source-and-validation) records the
shared final historical build, CTest, and lint results. No runtime checks were
rerun as part of this document cleanup.
