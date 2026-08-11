# Complete the Python API for QCO decision-diagram execution

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Python users can already build, simulate, and sample QCO programs with decision
diagrams, but cannot supply concrete values for symbolic entry arguments and
cannot sample from an existing input state. After this work, each Python DD API
accepts entry-argument bindings and both sampling APIs optionally accept an
initial `VectorDD`. A user can therefore execute parameterized gates and dynamic
qtensor inputs without dropping to C++.

## Progress

- [x] (2026-08-11 09:30Z) Identified the missing forwarding in
      `bindings/mlir/register_mlir.cpp` and the existing C++ `DDBindings`
      overloads.
- [x] (2026-08-11 09:36Z) Designed a mapping from zero-based entry-argument
  indices to Python bool, int, or float values.
- [x] (2026-08-11 09:39Z) Implemented conversion and forwarded bindings through
  all DD APIs.
- [x] (2026-08-11 09:39Z) Exposed initial-state sampling through both Python
  sampling APIs.
- [x] (2026-08-11 09:42Z) Added seven passing Python tests and regenerated the
  MLIR stub.

## Surprises & Discoveries

- Observation: Generated `.pyi` files must be regenerated with the repository's
  `stubs` Nox session and must not be edited manually. Evidence: `AGENTS.md`.
- Observation: `qco::QubitType` is provided by `QCOOps.h`, not by the DD utility
  header. Evidence: the first binding build failed until the direct include was
  added.

## Decision Log

- Decision: Keep the C++ `DDBindings = DenseMap<Value, Attribute>` contract and
  translate a Python mapping at the binding boundary. Rationale: MLIR `Value`
  objects are implementation details that Python callers should not need to
  construct merely to bind entry arguments. Date/Author: 2026-08-11, Codex.

## Outcomes & Retrospective

Python callers can now bind symbolic scalar inputs and dynamic qtensor extents
with `bindings={argument_index: value}`. Both sampling APIs accept an optional
`initial_state`. Existing call forms remain compatible, and all seven focused
Python tests pass.

## Context and Orientation

`bindings/mlir/register_mlir.cpp` defines nanobind functions on `mqt.core.mlir`.
`mlir/include/mlir/Dialect/QCO/Utils/DDFunctionality.h` defines the C++ APIs and
`DDBindings`. A binding associates an entry-block argument with an MLIR
`Attribute`; scalar attributes provide bool, integer, index, or floating values,
while an integer attribute bound to a dynamic qtensor argument provides its
extent. `test/python/test_qco_dd.py` owns end-to-end Python coverage.

## Plan of Work

Add a binding conversion helper in `bindings/mlir/register_mlir.cpp`. The
Python-facing mapping will use non-negative entry-argument indices as keys and
Python bool, int, or float values as values. The helper will inspect the entry
function argument type and construct the matching MLIR attribute. It will reject
unknown indices, negative dynamic qtensor extents, and type mismatches with a
clear `ValueError`. Add an optional keyword-only `bindings` parameter to
`build_functionality`, `simulate`, `sample`, and `sample_with_classics` and
forward the converted map to C++.

Add overload-friendly Python arguments for `sample` and `sample_with_classics`
so callers may provide an optional initial state. Preserve the existing
zero-state call form and reference-consumption semantics. Update docstrings and
regenerate `python/mqt/core/mlir.pyi`. Add Python tests for a symbolic rotation,
scalar-controlled flow, a dynamic qtensor extent, invalid bindings, and sampling
from a nonzero state.

## Concrete Steps

From the repository root, edit the binding and test files with focused patches.
Build the extension with:

    ./.agent/run.sh cmake --build --preset release

Run focused Python tests with:

    ./.agent/run.sh uv run --no-sync pytest test/python/test_qco_dd.py

Regenerate stubs with:

    ./.agent/run.sh uvx nox -s stubs

Finally run:

    ./.agent/run.sh uvx nox -s lint

## Validation and Acceptance

Acceptance requires Python tests proving exact matrix or deterministic sampling
behavior for bound symbolic inputs and a dynamic qtensor argument. Existing
calls without `bindings` or `initial_state` must remain source compatible. The
generated stub must describe the new keyword arguments. The focused Python test
file and lint session must pass.

## Idempotence and Recovery

Builds, tests, and stub generation are repeatable. If stub generation modifies
unrelated generated files, inspect the diff and retain only changes caused by
the binding signatures. Never hand-edit the generated stub.

## Artifacts and Notes

The pre-change Python wrappers call C++ without a `DDBindings` argument at
`bindings/mlir/register_mlir.cpp` around the `build_functionality`, `simulate`,
`sample`, and `sample_with_classics` definitions.

## Interfaces and Dependencies

Use nanobind types already included by `bindings/mlir/register_mlir.cpp` and
MLIR builtin attribute constructors. Do not introduce another serialization
format or dependency. The public Python signatures must accept an optional
mapping from `int` to `bool | int | float`, plus an optional `VectorDD` for the
sampling functions.

Revision note: Initial plan created to cover symbolic bindings, dynamic qtensor
extents, and input-state sampling in one coherent Python API change.
