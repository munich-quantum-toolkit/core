# Refocus QCO DD execution and sampling

Status: historical implementation record.

## Goal and scope

PR #2077 exposes DD building, simulation, and sampling of declared CBits or the
final basis state. QC coalesces static references; QCO owns one root per index.

## Constraints

- `Operation::fold` mutates and cannot coalesce siblings; hoisting plus CSE can.
- Greedy rewriting deletes dead operations; preflight precedes static rewriting.
- Cleanup changes fallback width; split histograms lose correlation. Do neither.

## Decisions

- hoist pure roots, run CSE, and cache builder indices. Rationale: shared MLIR
  replaces private logic.

- with an MQT entry point, every `qco.static` belongs to its entry block with
  unique indices; helpers take arguments. Verify transforms on both sides.
  Rationale: one QCO ownership boundary.

- `sample` uses conventional count-string order: the last returned CBit register
  comes first, and each register is MSB-first. This avoids adapter reordering.

- without CBit results, `sample` uses `measureAll`; mixed or undefined outputs
  fail. Loops use widened `APInt` and one 10,000-step budget.

## Outcome and validation

OpenQASM avoids duplicate roots; QC cleanup and conversion normalize other IR.
The private coalescer is gone. Python passes 6 focused and 3,143 matrix tests;
CTest passes 4,042 tests. Lint, C++ lint, stubs, and builds pass.

## Code and ownership

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

## Acceptance

Tests cover root normalization and ownership, unchanged fallback width, output
ordering, both sampling paths, loop budgets, and balanced DD references.

## Follow-ups

PR #2078 owns bindings, more scalar types, qtensors, and `scf.while`. PR #2079
owns multi-block control flow, budgeted block transitions, and DD-native
deallocation. Neither restores histories, supplied-state sampling, generic
folding, per-loop caps, or simulator canonicalization.
