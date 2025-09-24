# Quantum Dialect Revamp — Design Rationale and Background

Author: Junie (auto-generated)
Date: 2025-09-23
Status: Supplementary design rationale to the implementation plan
Scope: Rationale for the revamped MLIR quantum dialect architecture under mlir/

## Abstract

This document motivates and explains the major design decisions in the planned revamp of the MQTRef and MQTOpt MLIR dialects for hybrid classical–quantum computing. The goals are to improve composability, analysis, and ergonomics by introducing first-class modifier ops (controls, inverse, power), a sequence op, a unified UnitaryOpInterface, and a lightweight UnitaryExpr support library. The rationale emphasizes alignment with idiomatic MLIR, predictable normalization via canonicalization, and performance-conscious design for common 2×2 and 4×4 unitaries used in compiler transformations.

## 1. Background and Objectives

Current challenges:

- Controls and other modifiers are embedded into each unitary gate, complicating analysis and normalization.
- Builder ergonomics are cumbersome; textual IR lacks compact sugar for controlled variants.
- Two dialects (reference semantics vs value semantics) diverge where interfaces should be uniform.

Primary objectives:

- Treat composition as a first-class concern (explicit modifier and sequence ops).
- Unify analysis via a single interface (UnitaryOpInterface) with dialect-adaptive in/out views.
- Preserve performance for frequent 2×2/4×4 unitaries without adding heavy dependencies.
- Keep the IR normalized and ergonomic through canonicalization and parser/printer sugar.

## 2. Why Explicit Modifiers and Sequences?

### 2.1. Explicit inv/ctrl/negctrl/pow as wrapper ops

- Composability: Modifiers become uniform constructs that can wrap either single gates or entire sequences.
- Laziness: Evaluation is deferred; transformations can rewrite at the IR level without materializing matrices.
- Normal forms: Canonicalization can enforce a stable modifier order (inverse innermost, controls outermost), ensuring predictable patterns and avoiding rewrite loops.

Illustration (MLIR):

```mlir
// Before (implicit modifiers baked into gates)
// mqtref.x %q ctrl(%c)

// After (explicit wrappers)
mqtref.ctrl(%c) { mqtref.x %q }
```

### 2.2. Sequence op

- Expressive power: A seq region contains only unitary operations (including wrapped sequences). It captures common subcircuits as first-class objects.
- Implicit terminator: Smoother user experience and more idiomatic MLIR text.
- Analysis: The UnitaryExpr for a sequence is the product of child unitaries in program order.

Example:

```mlir
mqtref.seq {
  mqtref.h %q0
  mqtref.ctrl(%c) { mqtref.x %q0 }
}
```

## 3. Interface Unification Across Dialects

- MQTRef (reference semantics) and MQTOpt (value semantics) share the same high-level queries via UnitaryOpInterface.
- Dialect-adaptive accessors:
  - getTargetOperands(), getPosControlOperands(), getNegControlOperands() — always available.
  - getTargetResults(), getPosControlResults(), getNegControlResults() — meaningful for MQTOpt; empty for MQTRef.
  - Aggregates: getAllOperandQubits() and getAllResultQubits() (the latter empty in MQTRef).
- Consistency: Clients (passes, analyses) no longer branch on the concrete dialect; they check hasTargetResults().

## 4. Linear Controls in Value Semantics

- In MQTOpt, all qubits—including controls—are value-semantics and should be treated linearly.
- Modifiers and sequences consume and produce fresh SSA values for controls even if the control action is semantically identity.
- Benefits: Enforces single-use constraints, simplifies dataflow reasoning, and aligns with classical SSA-style optimizations.

Example:

```mlir
// Controls are threaded linearly in mqtopt
%q1, %c1 = mqtopt.ctrl(%c0) { mqtopt.x %q0 }
```

## 5. Hermitian and Diagonal Traits

- Hermitian (self-inverse) gates: I, X, Y, Z, H (and possibly others after validation). Canonicalization shortcut: inv(G) -> G.
- Diagonal gates (in computational basis): I, Z, S, Sdg, T, Tdg, Rz, P/Phase, RZZ. These enable commuting analyses and targeted lowers.
- Both are marker traits; they carry no runtime cost and inform canonicalization and analysis.

## 6. Canonicalization Strategy and Modifier Ordering

- Ordering: inverse (innermost) — power — controls (outermost). Rewriters maintain this normal form.
- Selected canonicalization:
  - inv(inv(X)) -> X
  - inv(pow(X, k)) -> pow(inv(X), k)
  - pow(X, -k) -> pow(inv(X), k)
  - inv(ctrl(X)) -> ctrl(inv(X))
  - inv(baseGate) -> baseGateInverse (parameters adjusted)
  - ctrl/negctrl with empty lists are removed; adjacent modifiers of the same kind are merged (no dedup or reorder of controls).

Examples:

```mlir
// inv(pow(X, k)) -> pow(inv(X), k)
mqtref.inv mqtref.pow(3) { mqtref.x %q }
// -> mqtref.pow(3) { mqtref.inv mqtref.x %q }

// pow(X, -k) -> pow(inv(X), k)
mqtref.pow(-2) { mqtref.h %q }
// -> mqtref.pow(2) { mqtref.inv mqtref.h %q }
```

## 7. Parameters: Prefer Static, Fold Constants

- Base-gate parameters should be static attributes whenever possible.
- If a dynamic operand is defined by a constant, fold it into static_params and adjust params_mask (or drop mask if all static).
- Benefits: Enables constant UnitaryExpr materialization and more effective downstream rewrites.

Examples:

```mlir
// Before: dynamic parameter
%pi2 = arith.constant 1.5707963267948966 : f64
mqtref.rx(%q) (%pi2)

// After: static attribute preferred
mqtref.rx(%q) (1.5707963267948966)
```

## 8. UnitaryExpr: Lightweight and Performant

- Tiny fixed-size complex matrices (2×2, 4×4) plus a compact expression graph (Mul, Adj, Pow, Control, Const, Param, Trig, Exp, Embed).
- No external dependencies; leverages LLVM/MLIR support and std::complex.
- Materialization to DenseElementsAttr for any arity (no artificial limit), with fast paths for 2×2/4×4.
- Intended usage: transformation and conversion passes; not a runtime simulator.

Example (C++ sketch):

```cpp
UnitaryExpr u = UnitaryExpr::Const(Mat2::X());
UnitaryExpr v = UnitaryExpr::Pow(u, 3);
UnitaryExpr w = UnitaryExpr::Adj(v);
auto maybeDense = w.materializeIfStatic(ctx); // DenseElementsAttr when all params static
```

## 9. Sequence Unitary and Footprint

- The UnitaryExpr of a seq is the product of child UnitaryExpr in program order (M_n … M_2 M_1).
- Sequences expose their overall qubit footprint only via getAllOperandQubits()/getAllResultQubits(). The notions of "controls vs targets" are not meaningful for seq as a whole.

Example:

```mlir
// The sequence's unitary is the product of the inner unitaries in program order
mqtref.seq {
  mqtref.h %q
  mqtref.rz(%q) (0.5)
}
```

## 10. Ergonomics: Builders and Sugar

- Generic builder template that dispatches using trait arities; layered with generated overloads per base gate.
- Parser/printer sugar provides concise forms like cx/cz/ccx/ccz and mcx/mcz/mcp(theta).

Examples:

```mlir
// Sugar for controlled gates
mqtref.ccx %c0, %c1 %t
mqtref.mcp(1.234)(%c0, %c1) %t

// Nested modifiers, implicit terminators
mqtref.ctrl(%c) { mqtref.inv mqtref.rx(%q) (3.14159) }
```

## 11. Alternatives Considered

- Single shared base-gate dialect vs per-dialect mirrored base gates: A shared dialect reduces op duplication but clashes with semantics (mqtref operands-only vs mqtopt operands+results), forcing optional results or wrappers and complicating verifiers/canonicalizations; modifiers/seq still require dialect-specific threading. Decision: keep per-dialect base gates; minimize duplication via shared traits/interfaces, TableGen catalogs, and generated builders; use symbol-based unitary.def/apply for custom gates.
- Embedding modifiers into every base gate: rejected due to poor composability and normalization challenges.
- Heavy linear-algebra backends for unitary composition: rejected to avoid dependencies and preserve compile-time performance.
- Treating controls in mqtopt as pass-through results: rejected; linear threading yields better dataflow properties and SSA adherence.

## 12. Scope Boundaries

- QubitRegister handling is out of scope; a separate PR will replace it with memref. All new design elements operate solely on Qubit.
- No temporary helper conversion layers; all affected code will be updated directly.

## 13. Expected Impact

- More predictable, analyzable IR with explicit composition mechanisms.
- Better usability for both textual IR authors and C++ clients through sugar and builders.
- Robust foundations for optimization passes (normalization, propagation, constant folding) and dialect conversions.

## 14. References and Further Reading

- MLIR Interfaces and Traits (mlir.llvm.org)
- QIRO: A Static Single Assignment based Quantum Program Representation for Optimization (doi:10.1145/3491247)
- Project plan: docs/mlir/quantum-dialect-revamp-plan.md
