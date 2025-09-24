# Tracking: Quantum Dialect Revamp (Self‑Contained)

This issue tracks the design and implementation of a revamped MLIR quantum dialect stack for MQT. It is self‑contained and does not assume any other local files exist.

## Context and Goals

We maintain two quantum dialects:

- mqtref — reference/memory semantics
- mqtopt — value semantics

Problems with the current setup:

- Modifiers (controls, inverse/adjoint, power) are baked into each gate op.
- Composition is clumsy; interfaces are not uniform across dialects.
- Builders are verbose; parser/printer sugar is limited.

High‑level goals:

- Make composition first‑class: explicit IR ops for modifiers that can wrap gates or sequences and lazily defer evaluation.
- Provide a single, idiomatic UnitaryOpInterface for analysis across both dialects.
- Dramatically improve ergonomics via a generic builder template and parser/printer sugar (e.g., `cx`, `ccx`, `mcx`, `mcp(theta)`).
- Keep dialects minimal and uniform; push optional features into wrappers/regions.
- Prefer MLIR‑native infrastructure (traits, interfaces, canonicalization, folds, symbol tables).
- QubitRegisters have been replaced by memrefs; only Qubit is used here.

## Architecture Overview

- Common layer
  - Traits and utilities shared by both dialects (TargetArity, ParameterArity, NoControl, Hermitian, Diagonal).
  - A unified UnitaryOpInterface with dialect adapters (exposes both operands and results for value semantics).
  - Parser/printer utilities for modifiers, parameters, sequences, and sugar.
  - A lightweight UnitaryExpr library is planned for transformation/conversion passes (no external dependencies, fast 2×2/4×4 paths). Its concrete implementation is deferred to execution of the plan.
- mqtref dialect (reference semantics)
  - BaseGate ops with only targets and parameters (no embedded modifiers/controls).
  - Resource ops (alloc/dealloc, static qubit), reset, measure.
  - Sequence and Modifier wrapper ops.
- mqtopt dialect (value semantics)
  - Mirrors mqtref base set but with value results for targets.
  - Sequence and Modifier wrapper ops with linear threading of all qubits, including controls.

## Marker Traits (Common)

- Hermitian — self‑inverse gates (e.g., I, X, Y, Z, H). Canonicalization short‑circuit: `inv(G) -> G`.
- Diagonal — diagonal in computational basis (e.g., I, Z, S, Sdg, T, Tdg, Rz, Phase, RZZ). Enables commutation/aggregation analyses.

## Types

- Qubit is the only type used by gates in this revamp.
- QubitRegister is assumed absent and will be handled in a separate PR.

## Base Gate Ops (Both Dialects)

- Minimal, uniform base gates with fixed target arity (TargetArity<N>) and parameter counts (ParameterArity<N>). No control operands or modifier attributes.
- Parameters prefer static attributes; constant folding turns dynamic constant operands into static attributes. No mask for `pow`.
- Each base gate provides its UnitaryExpr, and can materialize a DenseElementsAttr when fully static (no arity limit by design).
- Base-gate placement decision: keep base gates per dialect (mqtref and mqtopt); do not introduce a shared executable base-gate dialect. Rationale: semantics mismatch (operands-only vs operands+results) would force optional results/wrappers and complicate verifiers/canonicalizations; duplication is minimized via shared traits/interfaces, a Common TableGen catalog, and generated builders. For custom gates, use symbol-based unitary.def/apply.

## Modifiers

Explicit wrapper ops that accept a single unitary op (gate or sequence) in a region with implicit terminator:

- inv — adjoint of the child unitary
- ctrl / negctrl — positive/negative controls (variadic)
- pow(r) — real exponent; prefer static attribute

Canonicalization and folds:

- inv(inv(X)) -> X
- inv(pow(X,k)) -> pow(inv(X), k)
- inv(ctrl(X)) -> ctrl(inv(X))
- pow(X,1) -> X; pow(X,0) -> identity
- pow(pow(X,a), b) -> pow(X, a\*b)
- pow(X, -k) -> pow(inv(X), k)
- inv(baseGate) -> baseGateInverse (each base gate defines its inverse; parameters updated)
- Controls are outermost modifiers; inverse is innermost. Rewriters normalize order.
- Controls are not deduplicated or reordered by canonicalization; duplicates are rejected by verifiers.
- In mqtopt, controls are linear SSA values and must be consumed/produced (not forwarded unchanged).

## Sequence Op

- `seq` is a region (single block, implicit terminator) containing only UnitaryOpInterface ops (including modifiers and nested sequences).
- Canonicalization: inline nested sequences; remove empty sequences; fuse adjacent sequences; move `inv` across sequences by reversing order and adjointing.
- Querying: `getAllOperandQubits()` and `getAllResultQubits()` (mqtopt) expose the overall set of qubits the sequence acts on. `seq` itself does not distinguish controls vs targets.
- The UnitaryExpr of a `seq` is the product of child unitaries in program order (M_n … M_2 M_1).

## Unified UnitaryOpInterface (Common)

Idiomatic methods (dialect‑adaptive):

- Identification/meta: `getIdentifier()`, `getNumTargets()`, `getNumControls()`, `hasControls()`, `isSingleQubit()`, `isTwoQubit()`, `hasParams()`, `hasDynamicParams()`, `isOnlyStaticParams()`.
- Operands/results: `getTargetOperands()`, `getPosControlOperands()`, `getNegControlOperands()`, `hasTargetResults()`, `getTargetResults()`, `getPosControlResults()`, `getNegControlResults()`.
- Aggregates: `getAllOperandQubits()`, `getAllResultQubits()`.
- Unitary: `getUnitaryExpr()`.
- Value‑semantics verification ensures in/out segment sizes match for targets and controls.

## Arbitrary Unitaries and Gate Definitions

Not all operations are covered by base ops. The IR supports user‑defined unitaries in two complementary ways. Definitions are declared once (as symbols) and instantiated many times.

- Matrix‑defined unitary (feasible for small n): supply a 2^n×2^n complex matrix attribute. Target arity n is inferred from the matrix shape. Parameters may be modeled via attributes; prefer static folding. No arity limit on materialization.
- Composite‑defined unitary: supply a region with a `seq` of known gates and modifiers; formal parameters (f64) and formal qubit arguments are allowed. This covers parameterized gates universally.

IR surface (per dialect; names are idiomatic to MLIR):

- `mqtref.unitary.def` (SymbolOpInterface, optional matrix attr, optional body region; body contains `mqtref.seq`).
- `mqtref.unitary.apply @sym (params?)` on target qubits; no results (reference semantics). Modifiers can wrap apply.
- `mqtopt.unitary.def` (mirror using value‑semantic ops; body must thread values linearly).
- `mqtopt.unitary.apply @sym (params?)` on target qubit values; yields updated target values.

Verification:

- Matrix shape must be square and power‑of‑two; infer n. If both matrix and region are provided, check consistency when possible.
- Regions must contain only UnitaryOpInterface ops and correct argument/result counts.

Canonicalization:

- Inline small composite definitions at call sites when profitable to expose further simplifications.
- Materialize matrix‑defined unitaries to UnitaryExpr on demand.

## Builders and Parser/Printer Sugar

- Generic builder template that accepts targets as Span and parameters as a ParamSet, dispatching based on trait arities; layered overloads provide ergonomic APIs: `x(q)`, `rx(q, theta)`, `cx(c, t)`, `ccx(c0, c1, t)`, `mcx(ctrls, t)`, `mcp(ctrls, t, theta)`.
- Parser sugar for: `cx`, `cz`, `ccx`, `ccz`, `mcx`, `mcz`, `mcp(theta)`; nested forms like `ctrl(%c) inv rx(%q) (pi/2)` print and round‑trip.

## Testing Strategy and Robustness

- TDD first: write failing tests (LIT and C++) before implementation; keep them small and focused.
- Prefer round‑trip `mlir-opt` tests; normalize with `-canonicalize -cse` when textual formatting is not the target of the test.
- Use `-split-input-file` to group many small tests; use `-verify-diagnostics` for negative cases with `// expected-error`.
- Avoid brittle SSA numbering in FileCheck; anchor on op names/attributes and use `CHECK-LABEL`, `CHECK-SAME`, `CHECK-DAG` appropriately. Use `-mlir-print-op-generic` where necessary for stability.
- Add C++ unit tests that parse/build IR programmatically and verify via Operation::verify() and pass runs; this catches checkstring mistakes by asserting structural invariants.

## Verification and Canonicalization Summary

- Base gates: enforce TargetArity and ParameterArity; static parameter preference and folding.
- Modifiers: enforce single child unitary; linear control threading in mqtopt; pow has no mask.
- Sequence: implicit terminator; only unitary ops allowed.
- Normal form: controls outermost; inverse innermost; merges of adjacent like‑modifiers; empty controls dropped; nested sequences flattened.
- Hermitian/Diagonal traits guide short‑circuits and analyses.

## Passes and Conversions

- Normalization, ControlPushdown, AdjointPropagation, ParamConstFold passes.
- mqtref → mqtopt conversion: map base gates/modifiers; `seq` with implicit terminator; thread controls linearly and produce fresh SSA results (including for controls).
- mqtopt → mqtref conversion: erase value results; controls become operands only; maintain modifier/sequence structure.

## Milestones and Tasks

Use this checklist to plan and track progress.

M1 — Foundations (Common)

- [ ] Unified UnitaryOpInterface (Common).
- [ ] Parser/Printer utilities for params, modifiers, sequences, sugar.
- [ ] Marker traits: Hermitian, Diagonal.
- [ ] UnitaryExpr library skeleton planned (implementation deferred to execution of the plan).

M2 — mqtref Base + Modifiers + Sequence

- [ ] Redefine base UnitaryOp (no embedded controls).
- [ ] Implement modifiers: ctrl/negctrl/inv/pow with implicit terminators.
- [ ] Implement seq with implicit terminator; inliner + canonicalization.
- [ ] Implement unitary.def (symbol) and unitary.apply (instantiation).
- [ ] Update std gate defs and assembly formats; add sugar (cx/cz/ccx/ccz/mcx/mcz/mcp).
- [ ] LIT tests: ops, parsers, canonicalization, unitary.def/apply.

M3 — mqtopt Mirror

- [ ] Redefine base gate ops with value results for targets.
- [ ] Modifiers with linear control threading; seq with implicit terminator.
- [ ] Implement unitary.def (symbol) and unitary.apply (instantiation) with value semantics.
- [ ] Interface/verifier updates for in/out segment size equality.
- [ ] LIT tests mirroring mqtref, including unitary.def/apply.

M4 — Builders and Ergonomics

- [ ] Generate CircuitBuilder helper API (generic template + overloads).
- [ ] Parser/printer sugar for compact modifier nesting and controlled variants.
- [ ] C++ smoke tests for builders.

M5 — Passes and Conversions

- [ ] Implement Normalization, ControlPushdown, AdjointPropagation, ParamConstFold.
- [ ] Update conversions mqtref↔mqtopt for modifiers/sequences and linear controls.
- [ ] Tests for conversions and passes.

M6 — Documentation and Polishing

- [ ] Ensure all ODS docstrings have fenced MLIR examples.
- [ ] Update user docs and examples; final test stabilization.

## Acceptance Criteria

- [ ] Dialects compile; LIT tests cover base gates, modifiers, sequences, unitary.def/unitary.apply, and sugar.
- [ ] Unified UnitaryOpInterface used across unitary‑capable ops, exposing operand/result views as appropriate.
- [ ] Arbitrary unitary support: symbol‑based unitary.def with matrix‑ and composite‑defined forms; unitary.apply in both dialects; modifiers wrap applications.
- [ ] Builder API (template + overloads) with sugar for cx, cz, ccx/ccz, mcx/mcz/mcp.
- [ ] Canonicalization suite maintains normal form; mqtopt controls are linearly threaded and not forwarded unchanged.
- [ ] Conversions mqtref↔mqtopt handle modifiers/sequences and control threading.

## Examples (MLIR)

mqtref, modifiers and sugar:

```mlir
// Single‑controlled rotation
mqtref.crx %c, %t (1.234)

// Double‑controlled Toffoli
mqtref.ccx %c0, %c1 %t

// Multi‑controlled X and P(theta)
mqtref.mcx(%c0, %c1, %c2) %t
mqtref.mcp(1.234)(%c0, %c1) %t

// Nested modifiers
mqtref.ctrl(%c) mqtref.inv mqtref.rx(%q) (3.14159)
```

mqtopt, sequence and linear controls:

```mlir
%q0 = mqtopt.allocQubit
%c  = mqtopt.allocQubit
mqtopt.seq {
  %q0_1 = mqtopt.h %q0
  // ctrl threads control linearly: consumes %c, yields %c_1
  %q0_2, %c_1 = mqtopt.ctrl(%c) { mqtopt.x %q0_1 }
}
```

User‑defined unitaries:

```mlir
// mqtref: define via region and apply
mqtref.unitary.def @H2 {
  ^entry(%q: !mqtref.Qubit):
    mqtref.seq {
      mqtref.rx(%q) (1.5707963267948966)
      mqtref.z %q
    }
}
mqtref.unitary.apply @H2 %q0
mqtref.ctrl(%c) { mqtref.unitary.apply @H2 %q1 }

// mqtopt: parameterized composite definition and application
mqtopt.unitary.def @CRZ(%theta: f64) {
  ^entry(%q0: !mqtopt.Qubit, %q1: !mqtopt.Qubit):
    mqtopt.seq {
      %q1_1 = mqtopt.rz(%q1) (%theta/2.0)
      %q0_1, %q1_2 = mqtopt.cnot %q0, %q1_1 : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit)
      %q1_3 = mqtopt.rz(%q1_2) (-%theta/2.0)
      %q0_2, %q1_4 = mqtopt.cnot %q0_1, %q1_3 : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit)
    }
}
%q0_1, %q1_1 = mqtopt.unitary.apply @CRZ(%q0, %q1) (3.14159)
%q0_2, %q1_2 = mqtopt.ctrl(%c) { mqtopt.unitary.apply @CRZ(%q0_1, %q1_1) (0.25) }
```

## Notes

- Controls must be outermost modifiers; inverse innermost; power in between. Rewriters enforce normalized order.
- Controls are not deduplicated by canonicalization; verifiers reject duplicates.
- QubitRegister is assumed absent in this revamp; only Qubit is used. A separate PR will replace any remaining register usage with memref.
