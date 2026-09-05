# Flatten reusable quantum functions for QIR output

This ExecPlan follows `.agent/PLANS.md`. Keep its progress, discoveries,
decisions, and outcome current. Run commands from the repository root.

## Purpose

QIR is a flat output format in the compiler: its lowering does not preserve QC
or QCO function calls. Make Base and Adaptive QIR generation accept reusable
unitary helpers without changing the compact QC/QCO representation used by other
outputs. Calls nested in quantum modifiers must retain their phase and modifier
semantics.

## Progress

- [x] (2026-09-04) Rebase onto current `main` and the finalized jeff function
      layer, adapting the implementation to the split Programs/Pipeline
      libraries.
- [x] (2026-09-04) Reuse MLIR's inliner and existing phase, modifier, and
  canonicalization passes for flat QIR preparation.
- [x] (2026-09-04) Register Func and LLVM inliner extensions for compiler-owned
  and caller-owned contexts.
- [x] (2026-09-04) Keep early QCO inlining exclusive to coordinated QIR output.
- [x] (2026-09-04) Preserve direct, CLI, jeff-imported, and both-profile tests.
- [x] (2026-09-04) Apply the independent QIR/MLIR specialist review: remove a
  duplicate textual-inliner check and defer an inert OpenQASM test switch.
- [x] (2026-09-04) Build the compiler and CLI and pass all 161 compiler tests.
- [ ] Run repository lint on the final stack.

## Discoveries

Plain reusable calls are illegal at the QC-to-QIR boundary. A helper containing
a global phase under an integral power needs more than inlining: phase
normalization extracts and scales the phase, modifier unrolling distributes the
supported operations, and canonicalization folds the remaining one-operation
modifier.

A same-wire composite helper under a power must be exposed while it is still QCO
so existing QCO synthesis can reduce it safely. Distributing a power over a
general noncommuting sequence would be incorrect. Early QCO inlining therefore
belongs only to QIR-bound coordinated pipelines, before target compilation or
the default QCO optimization pipeline.

MLIR's stock inliner depends on promised Func and LLVM interfaces. Compiler
contexts and contexts adopted by typed programs must install those extensions;
otherwise the public textual `inline` pipeline can abort before examining QC or
QCO operations.

Current `main` separates context/parsing code in `Programs.cpp` from conversion
and coordinated compilation in `Pipeline.cpp`. Inliner-extension registration
stays with context ownership; QIR preparation and output routing stay in the
pipeline library.

## Decisions

Use `createInlinerPass`, `NormalizeGlobalPhases`, `UnrollModifiers`, and the
canonicalizer. Do not add a QIR-specific call lowering, custom call graph, or
new synthesis pass.

Install standard inliner extensions in `createCompilerContext` and when a typed
program adopts a caller-owned context. This makes the public compiler program
contract independent of who constructed the context.

Inline all QCO calls early only for Base or Adaptive QIR output. Preserve calls
for QCO, QC, OpenQASM, and jeff output. The direct `QCProgram::intoQIR` path
uses the common QC preparation pipeline immediately before profile lowering.

Keep unsupported composite powers fail-closed. The compiler need not invent an
unsafe algebraic rewrite merely to accept a hypothetical program.

## Scope

`mlir/lib/Compiler/Programs.cpp` owns compiler-context extension registration.
`mlir/lib/Compiler/Pipeline.cpp` owns typed and coordinated QIR preparation.
`mlir/lib/Support/Passes.cpp` exposes the shared QC-to-QIR preparation sequence.
`mlir/tools/mqt-cc/mqt-cc.cpp` mirrors the same boundaries for its direct MLIR
pipeline.

The compiler regression covers a phase-bearing unitary helper under nested power
and control, caller-owned textual inlining, early QCO exposure, and a
binary-restored jeff helper through both QIR profiles. The existing OpenQASM
program matrix continues to exercise flat QIR generation from production
frontend inputs.

## Validation

Configure and build the compiler and CLI:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler mqt-cc -j4

Run the reusable-function regressions and full compiler test binary:

    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='CompilerPipelineTest.*QIR*:*OpenQASM*'
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

The nested helper must produce Base and Adaptive QIR with one correctly scaled
relative-phase call. Caller-owned contexts must run the textual inliner without
an abort. Coordinated QIR output must expose QCO helpers before synthesis, while
non-QIR outputs retain them.

Finish once on the final stack with:

    uvx nox -s cpp-lint
    uvx nox -s lint

Hosted CI is separate evidence and counts only after the final rewritten branch
is pushed. Builds, tests, and extension registration are repeatable. This plan
does not authorize unrelated remote changes.

## Outcome

Focused validation passes; final stack lint remains. The rebased implementation
uses native MLIR infrastructure and adds no call-specific QIR representation or
analysis. An independent specialist found the production implementation
idiomatic and removed only redundant or premature test scaffolding.
