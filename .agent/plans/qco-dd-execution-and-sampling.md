# Refocus QCO DD execution and sampling

This living ExecPlan follows `.agent/PLANS.md`; keep it current.

## Purpose / Big Picture

PR #2077 exposes DD building, simulation, and sampling of declared CBits or the
final basis state. QC coalesces static references; QCO owns one root per index.

## Progress

- [x] (2026-08-26) Implement direct, budgeted, output-aware execution and tests.
- [x] (2026-08-29) Rebase; replace coalescing with reuse/CSE; enforce QCO roots.
- [x] (2026-08-29) Add boundary regressions and pass all local validation.
- [x] (2026-08-30) Rebase onto main after the static-qubit PR merged; drop its
  superseded commits while retaining DD-specific coverage.
- [x] (2026-08-31) Move the Python DD operations onto `QCOProgram` and require
  an RNG in the public C++ simulation API.

## Surprises & Discoveries

- `Operation::fold` mutates and cannot coalesce siblings; hoisting plus CSE can.
- Greedy rewriting deletes dead operations; preflight precedes static rewriting.
- Cleanup changes fallback width; split histograms lose correlation. Do neither.

## Decision Log

- Decision: hoist pure roots, run CSE, and cache builder indices. Rationale:
  shared MLIR replaces private logic. Date: 2026-08-29.
- Decision: with an MQT entry point, every `qco.static` belongs to its entry
  block with unique indices; helpers take arguments. Verify transforms on both
  sides. Rationale: one QCO ownership boundary. Date: 2026-08-29.
- Decision: `sample` uses conventional count-string order: the last returned
  CBit register comes first, and each register is MSB-first. This avoids adapter
  reordering. Date: 2026-09-02.
- Decision: without CBit results, `sample` uses `measureAll`; mixed or undefined
  outputs fail. Loops use widened `APInt` and one 10,000-step budget. Date:
  2026-08-26.

## Outcomes & Retrospective

OpenQASM avoids duplicate roots; QC cleanup and conversion normalize other IR.
The private coalescer is gone. Python passes 6 focused and 3,143 matrix tests;
CTest passes 4,042 tests. Lint, C++ lint, stubs, and builds pass.

## Context and Orientation

`mlir/lib/Dialect/QC/Builder/QCProgramBuilder.cpp` serves OpenQASM import;
`mlir/lib/Dialect/QC/IR/QubitManagement/DeallocOp.cpp` hoists QC roots;
`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` validates and converts them.
`mlir/lib/Dialect/QCO/IR/QCOUtils.cpp` owns QCO invariants; `Programs.cpp`
checks public transform boundaries; and
`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` executes single-block QCO.

The Python API is `program.build_functionality(dd_package) -> MatrixDD`,
`program.simulate(initial_state, dd_package, seed=0) -> VectorDD`, and
`program.sample(shots=1024, seed=0) -> dict[str, int]`. The public C++
simulation function always receives an RNG. Static sampling evolves once,
adaptive control runs per shot, and returned CBits share storage across calls.

## Milestones

### 1. Canonical QC roots

Cache, hoist, and CSE QC roots; conversion repeats this only after preflight.
Nested and duplicate inputs must lower to one entry-block root per index.

### 2. Strict QCO ownership

Reject duplicate indices and roots outside the MQT entry, but accept helper
arguments. Recheck transformations; never canonicalize during DD preparation.

### 3. Execute, sample, and prove the boundary

Retain exact values, direct execution, shared CBits, and one budget. Test
deferred and adaptive sampling, then run every repository check.

## Concrete Steps

From the repository root, run:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittest-qc-ir
    cmake --build --preset release --target mqt-core-mlir-unittest-qc-to-qco
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    cmake --build --preset release --target mqt-core-mlir-unittest-qco-utils
    uvx nox -s tests-3.14 -- test/python/test_qco_dd.py -q
    uvx nox -s stubs
    uvx nox -s lint
    uvx nox -s cpp-lint
    cmake --build --preset release
    ctest --preset release --output-on-failure
    uvx nox -s tests

Only the stubs session may regenerate `python/mqt/core/mlir.pyi`.

## Validation and Acceptance

Tests prove root normalization and ownership, unchanged fallback width, output
ordering, both sampling paths, loop budgets, and balanced DD references. Every
command exits zero. Refetch and verify the recorded SHA and signed commits, then
publish with exact `--force-with-lease`; never unguarded force.

## Downstream Boundaries

PR #2078 owns bindings, more scalar types, qtensors, and `scf.while`. PR #2079
owns multi-block control flow, budgeted block transitions, and DD-native
deallocation. Neither restores histories, supplied-state sampling, generic
folding, per-loop caps, or simulator canonicalization.

## Idempotence and Recovery

Builds and checks are repeatable; a backup preserves the old head. If the remote
advances, rebase and revalidate. Preserve unrelated changes and follow-up PRs.
