# Backport the DD implementation to v3.x

This plan follows `.agent/PLANS.md` and records the backport of main at
`220422824ba6df9983c63ce3d766303f2984c631` onto v3.x at
`0a9656a80da3dfd0466718d713e363ba85f89f01`.

## Purpose and scope

Wide circuit matrices must retain their global scale below the DD package's
ordinary numerical tolerance. The backport also adopts upstream DD performance
and correctness improvements without requiring LLVM or MLIR. The existing v3
circuit, operation, simulation, approximation, and Python entry points remain.
The separate garbage-reduction and QCEC zero-handling fixes are independent PRs.

## Progress

- [x] Establish the v3 baseline and reproduce the 83-qubit matrix collapse.
- [x] Import the compatible DD internals and upstream behavioral tests.
- [x] Preserve v3 adapters, control types, and wide gate argument validation.
- [x] Regenerate Python stubs and test public Python consumers.
- [x] Rebuild unchanged QCEC 3.10.2 and verify the issue and equivalent
      controls.
- [x] Review the implementation and add global-phase ownership coverage.
- [x] Complete final whole-file C++ lint and prepare the draft backport PR.

## Milestones and implementation

The DD headers in `include/mqt-core/dd/` and implementations in `src/dd/` now
use the upstream numeric storage, normalization, allocation, node tables,
compute caches, traversal, matrix operations, and state generation. Matrix root
weights have a separate exact index; internal coefficients retain tolerant
lookup. The upstream table growth, hashing, cache invalidation, direct matrix
builders, and element access improvements accompany this change.

The v3-only circuit adapters remain in `FunctionalityConstruction`,
`Simulation`, and `Operations`. The global-phase helper transfers ownership when
changing a root weight. Gate arguments still accept `qc::Qubit`, validate before
narrowing to internal DD indices, and use the existing IR control and
permutation types. Bindings keep these v3 interfaces while sharing the upstream
vector and matrix builders, including read-only and strided NumPy arrays.

Native regressions live in `test/dd/`; public binding tests live in
`test/python/dd/`. Existing v3 circuit integration tests remain. Tests of the
old real-table bucket layout are replaced with upstream numerical lookup and
garbage-collection tests because bucket placement changed.

## Surprises & Discoveries

The issue's independent phased Hadamard gates produce a root with real and
imaginary components `-2^-42` at 83 qubits. The old tolerant root lookup turns
both components into zero. The new regression fails on the v3 baseline and
preserves both components after the backport.

An imported Python signature referred to the v4-only `mqt.core.dd.Control`.
Review caught it; the signatures and generated stubs use
`mqt.core.ir.operations.Control`. A global-phase lifecycle probe also confirms
that the original helper loses ownership of the changed root, while the ported
helper survives forced collection and releases cleanly.

The DD changes alter C++ layouts. Downstream native extensions must be rebuilt.
Low-level APIs also follow upstream: `RealNumberUniqueTable::hash` becomes an
instance method, the obsolete `immortals` helper disappears, and invalid
numerical tolerances throw instead of being accepted by a `noexcept` setter.

## Decision Log

Use the upstream implementation directly where compatible, with v3 adapters at
its boundaries. Do not import the v4 circuit removal or LLVM-based consumers. Do
not combine terminal garbage reduction or QCEC's defensive zero detection with
this backport; each must be independently reviewable.

Retain semantic tests and add a direct 83-qubit matrix regression. Validate QCEC
against a source rebuild because the released wheel uses the previous C++
layout. No new performance claim is made without measurements.

## Validation

From the repository root, configure and build with `cmake --preset release` and
`cmake --build --preset release`, then run `ctest --preset release`. A supported
local release preset used Clang 23 and interprocedural optimization. The full
Python suite uses `uv run --no-sync pytest test/python`; it passed 575 tests
with three existing Qiskit-version skips. Run `uvx nox -s stubs` after binding
changes, `uvx nox -s cpp-lint -- HEAD^` after committing, and finish with
`uvx nox -s lint`.

For the QCEC integration, build unchanged QCEC 3.10.2 against this Core package.
Construct GHZ(n) and GHZ(n) followed by X on the final qubit, measure all
qubits, and transpile to `{cz, rz, sx, x}` at optimization level zero. Use only
the construction checker. At 82, 83, 90, 101, and 128 qubits, compare the two
translated circuits and compare the original GHZ with its translation, both with
ordinary inputs and with all inputs marked ancillary. Repeat both pairs with
ancillary inputs and partial equivalence enabled. All 30 checks return the
expected verdict: different circuits are non-equivalent; equivalent circuits are
equivalent, allowing global phase for full equivalence.

## Outcomes & Retrospective

The backport alone resolves the reported false verdicts and ancillary partial
comparison crash in rebuilt QCEC. Core's general terminal garbage-reduction bug
and QCEC's handling of future numerical collapse still need their separate
fixes. Human review and hosted CI remain pending; local validation is not a
claim about either.
