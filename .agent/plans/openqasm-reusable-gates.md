# Preserve reusable OpenQASM gates in QC

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current as required by
`.agent/PLANS.md`.

## Purpose / Big Picture

OpenQASM import used to expand every custom gate at every use, while export
accepted only one function. Preserve each custom definition as one private QC
function and each application as a call. Export those functions and calls back
to dependency-ordered OpenQASM gate declarations and applications. Nested and
repeated gates then stay compact across QC, QCO, OpenQASM, and QIR pipelines.

## Progress

- [x] Trace the typed frontend, importer, exporter, builder ABI, and QC/QCO
      round trips.
- [x] Preserve straight-line gates as `mqt.unitary` functions and structured
      gates as generic functions.
- [x] Export supported gate functions and calls in dependency order.
- [x] Add representation, rejection, deep-graph, and strict round-trip tests.
- [x] Fold the change into the existing OpenQASM changelog entry.
- [x] Rebase on the 2026-09-04 compiler and OpenQASM changes without regressing
      current classical-expression or structured-control support.
- [x] Delete the obsolete recursive frontend graph walk and dependency-depth
      ceiling now that definitions and calls remain compact.
- [x] Incorporate the post-rebase OpenQASM specialist review.
- [x] Run focused tests, compiler tests, C++ lint, and repository lint.

## Surprises & Discoveries

- The typed frontend already validates custom-gate dependencies and records
  whether a gate contains or transitively calls structured control.
- OpenQASM gate bodies may contain loops. Such functions cannot honestly carry
  `mqt.unitary`, so they use `func.call`; straight-line gates use `qc.call`.
- A source gate may be named `main`; the artificial entry symbol must then be
  renamed while `mqt.entry_point` remains the semantic entry marker.
- OpenQASM requires the rendered qubit operands of a gate application to be
  distinct, while verified QC permits aliases. Export therefore fails closed.
- Current `main` supports fixed-width classical expressions, snapshots, SCF
  results, and general while loops. Replaying the old exporter wholesale would
  have silently removed those capabilities.
- Folding the imported inclusive loop's `stop + 1` is required for immediate
  export: the exporter intentionally accepts only constant `scf.for` bounds.
- Forward gate references are rejected during semantic analysis, so valid gate
  dependencies only point backward. Rejecting the active gate at each call is
  sufficient to prevent recursion without a second graph traversal.

## Decision Log

- Decision: Preserve every parsed custom definition, including unused ones.
  Rationale: definitions are source program content and emitting each once is
  linear. Date/Author: 2026-09-03 / Codex.
- Decision: Use the function symbol as the gate name when legal, otherwise a
  collision-free generated name. Rationale: no source-name metadata is needed
  for the supported unique-name contract. Date/Author: 2026-09-03 / Codex.
- Decision: Use MLIR `CallGraph` and LLVM SCC traversal for callee-first order
  and recursion detection. Rationale: native infrastructure replaces a custom
  recursive traversal. Date/Author: 2026-09-03 / Codex.
- Decision: Remove the former 64-level dependency boundary. Rationale: calls no
  longer expand gate bodies, forward references are invalid, and self-recursion
  is rejected as the body is analyzed. Date/Author: 2026-09-04 / Codex.
- Decision: Reject classical storage and non-gate constructs inside exported
  gate functions. Rationale: gate definitions remain a pure, portable subset;
  entry-function capabilities stay unchanged. Date/Author: 2026-09-03 / Codex.
- Decision: Defer OpenQASM `def`, `extern`, calibration, arrays, returns, and
  quantum-register arguments. Rationale: they require a broader common-IR ABI,
  not speculative extensions to the gate ABI. Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

The rebased implementation adds no dialect operation, metadata, or external
dependency. It retains the newer exporter implementation and layers reusable
gates onto its current expression, CBit, and control-flow support. Specialist
review removed the obsolete recursive frontend graph validator, its arbitrary
depth limit, recursive capability inference, and dead gate-body flexibility.

## Context and Orientation

`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` consumes the typed
frontend's `program.gates`. It creates a private function for each definition,
caches it by definition, and emits calls at applications.

`mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp` selects the
`mqt.entry_point`, validates all remaining functions against the supported gate
ABI, orders them with MLIR's call graph, and emits definitions before the entry
body. `docs/mlir/OpenQASM.md` is the user-visible contract.

## Plan of Work

Import each definition once. Use `createUnitaryFunction` for a straight-line
definition and `createFunction` for a transitively structured definition. At a
custom application, concatenate parameters and qubits and call the cached
function. Charge the existing emission budget once per definition body and once
per use instead of recursively charging expanded bodies.

On export, accept private, defined, single-block functions whose arguments are
leading `f64` parameters followed by at least one scalar qubit and which return
nothing. Accept supported scalar expressions, unitary operations, calls, and
loops. Reject recursion, unresolved calls, classical storage, allocation,
measurement, reset, barriers, conditionals, switches, and arbitrary CFGs in gate
definitions. Emit SCCs in callee-first order and reject cyclic SCCs.

Keep tests centered on observable contracts: functions and calls survive import,
dependencies precede users on export, strict reimport succeeds, long graphs
remain compact, structured gates retain generic calls, floating loop expressions
retain their meaning, and invalid gate bodies or qubit aliasing fail directly.

## Concrete Steps

Run from the repository root:

    cmake --preset release
    cmake --build --preset release -j2
    build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target
    build/release/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittest-compiler
    uvx nox -s cpp-lint
    uvx nox -s lint

## Validation and Acceptance

Acceptance requires strict OpenQASM-to-QC-to-OpenQASM-to-QC round trips with one
function per custom definition and one call per application. Straight-line
functions carry `mqt.unitary`; structured functions do not. A long nonrecursive
graph remains linear and deterministic. Invalid signatures, bodies, recursion,
unresolved calls, and aliased rendered qubits fail with diagnostics. Focused
tests and lint pass on the final commit; hosted CI is separate evidence.

## Idempotence and Recovery

Configuration, builds, tests, and lint are repeatable. Translation failure
discards partial output or the partially built module. The rebase backup ref
retains the pre-update implementation; do not overwrite unrelated work.

## Artifacts and Notes

- Previous-stack evidence: 175 OpenQASM target tests, 187 QC translation tests,
  174 QC/QCO conversion tests, and repository lint passed.
- Current-main evidence: 198 QC translation tests, 185 OpenQASM frontend tests,
  161 compiler tests, and changed-file C++ lint passed after specialist
  feedback. Repository lint passed after restacking.

## Interfaces and Dependencies

Use existing `QCProgramBuilder`, `func::FuncOp`, `qc::CallOp`, `func::CallOp`,
`mqt::getEntryPoint`, `UnitaryOpInterface`, `SymbolTableCollection`, MLIR
`CallGraph`, and LLVM SCC traversal. `MLIRAnalysis` is the only added link to an
already available project dependency. Do not add an operation, attribute,
source-name side channel, or custom graph framework.

Revision note (2026-09-04): rebased on the redesigned compiler pipeline and the
expanded OpenQASM frontend/exporter, retained those capabilities, and removed a
speculative call-depth restriction.
