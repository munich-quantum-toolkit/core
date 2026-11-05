# Custom target compilation stages

Status: complete.

## Goal and scope

Support Python-only pass-order experiments with the existing
`QCOProgram.run_pass_pipeline` API. Expose target preparation, placement,
finalization, and the optimizations used by the canonical target pipeline.
Preserve logical layout tracking through separate calls, including terminal SWAP
elimination before placement. Keep generic pass behavior unchanged.

## Decisions

Use registered MLIR passes and named pipelines rather than per-pass Python
methods. Target-aware optimization is explicit through `use-target=true` and
uses `TargetEnvironmentAnalysis`. Reuse existing target preparation, native
synthesis, cleanup, and target projection rather than duplicating them in the
Python caller. Compiler-level stage registration must not introduce a dependency
from the support library back to the compiler pipeline library.

The runner keeps conservative layout invalidation by default. The worked
sequence explicitly preserves layout across stages that retain or update its
meaning. Searches start from a fresh logical input copy.

## Implementation

- [x] Expose preparation and finalization with a clear target metadata
      lifecycle.
- [x] Register placement, terminal SWAP elimination, and two-qubit fusion;
      expose target-aware modes of existing optimizations.
- [x] Demonstrate sparse and all-to-all targets with two native gate sets using
      public Python APIs, layout-aware equivalence, and final circuit metrics.
- [x] Validate native pass contracts, Python workflows, executable examples,
      generated references, and required repository/C++ lint.

## Validation

Existing compiler and transform tests cover native target semantics and negative
cases. `test/python/test_mlir.py` covers the Python handoff, with executable
guidance in `docs/mlir/target_compilation.md`. The regression fails on the
baseline because preparation is not registered. The updated wheel passes 472
Python tests, and the affected native tests pass 163 cases. Stub generation,
repository lint, and whole-file C++ lint pass. The C++ check covers all 18
modified translation units; uncommitted files omitted by the standard diff
selection were checked explicitly. The final lint fixes were rebuilt and their
affected native tests rerun.

Strict executable documentation and local link checks pass. This macOS setup
needed a short temporary path for simulator sockets, the configured LLVM SDK,
and a 120-second notebook timeout for variable import time. These were local
environment overrides. No performance improvement is claimed; this exposes
existing algorithms.
