# Custom target compilation stages

Status: complete.

## Outcome and scope

`QCOProgram.run_pass_pipeline` supports Python-only pass-order experiments with
target preparation, placement, finalization, and target-aware optimizations.
Separate calls retain logical layouts when requested, including terminal SWAP
elimination before placement. Generic pass behavior is unchanged. The executable
example in `docs/mlir/target_compilation.md` compares flat circuits; callers
must meet each pass's input requirements. No performance improvement is claimed.

## Decisions

Registered passes avoid per-pass Python bindings. `use-target=true` selects the
attached target through `TargetEnvironmentAnalysis`; omitting it preserves
generic behavior. `mlir/include/mqt/Dialect/QCO/Transforms/Passes.td` owns these
pass options and input contracts.

`mlir/lib/Compiler/TargetCompilation.cpp` reuses preparation, target projection,
cleanup, and native synthesis from the default compiler. Preparation saves the
execution environment in typed IR metadata, so copies and checkpoints retain it.
Successful finalization restores it; explicit environment attachment discards
it. Stages register at the compiler layer to avoid a dependency from the support
library back to the compiler pipeline library.

Finalization validates the saved environment before changing the program. Typed
IR alone does not reject disconnected targets, so restoring the attribute
without this check would accept invalid checkpoints.

The runner in `mlir/lib/Support/Passes.cpp` clears layout metadata by default.
Callers set `preserve_layout=True` only for passes that retain or update the
mapping. Failed execution can leave partial changes; searches use independent
copies and start from the logical source when changing targets.

## Validation

`mlir/unittests/Compiler/test_compiler_pipeline.cpp` covers stage lifecycle and
idle input order; transform tests cover native gate preservation and missing
targets. `test/python/test_mlir.py` checks two pass orders on sparse and
all-to-all targets with two gate sets, copy/checkpoint isolation, layout-aware
equivalence, and final circuit metrics. Run it with
`uv run --no-sync pytest test/python/test_mlir.py`. The staged regression fails
before this change because preparation is not registered.

The implementation passed 472 Python tests and 163 affected native tests before
the metadata tests were consolidated. The focused handoff cases and all 42 MQT
IR tests pass with the consolidated fixtures. See the
[development guide](../../AGENTS.md#build-and-validation) for repository lint,
C++ checks, executable documentation, and local link checks.
