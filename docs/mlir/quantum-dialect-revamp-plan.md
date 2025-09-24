# Quantum Dialect Revamp — Implementation Plan (Revision 5)

Author: Junie (auto-generated)
Date: 2025-09-24
Status: Design document, no IR code changes committed yet
Scope: MLIR dialect hierarchy under mlir/, affecting MQTRef and MQTOpt dialects and shared Common utilities

## 1. Context and Goals

We currently have two working quantum dialects:

- mqtref: reference/memory semantics
- mqtopt: value semantics

Both support a wide set of unitary "basis" gates and basic resources (qubit, alloc/dealloc, reset, measure). However, usage is cumbersome, builders are verbose, and the treatment of modifiers (controls, inverse/adjoint, powering) is embedded into each gate op, which complicates composition, normalization, and ergonomics.

This plan proposes a fundamental rework that:

- Makes composition first-class: modifiers are explicit, uniform IR constructs that wrap gates or sequences and lazily defer evaluation.
- Unifies analysis via interfaces: a single UnitaryOpInterface to query gate properties (targets, controls, parameters, unitary, etc.) without pattern matching op kinds, while accommodating the different semantics of mqtref and mqtopt.
- Greatly improves ergonomics: C++ CircuitBuilder helpers and parser/printer sugar to write MLIR like mqtref.ctrl(%c) mqtref.x %q or C++ like builder.x(q).ctrl(c).
- Keeps dialect-specific semantics minimal and uniform while sharing as much as possible between mqtref and mqtopt via Common ODS, traits, and support libraries.
- Remains analysis- and transformation-friendly and provides canonicalization and verifiers to maintain a normalized IR.

No backward compatibility is required. We do not change IR code yet; this document details the implementation plan.

## 2. High-level Architecture

2.1. Layering

- Common (shared between both dialects)
  - Traits and interfaces (TargetArity, ParameterArity, NoControl, UniqueSize/Index, plus new ones).
  - UnitaryOpInterface (single definition in Common with dialect adapters).
  - UnitaryExpr support library for symbolic unitary expressions and tiny fixed-size matrices.
  - Parser/printer utils for modifiers, sequences, parameters, and sugar.
- mqtref (reference semantics)
  - Types: Qubit only
  - BaseGate ops (no controls/inverse/pow; only targets and parameters).
  - Resource ops (allocQubit/deallocQubit, static qubit op), reset, measure.
  - Sequence and Modifier wrapping ops.
- mqtopt (value semantics)
  - Types: Qubit only
  - BaseGate ops (mirrors mqtref base set but value results for targets).
  - Resource/reset/measure counterparts with value semantics.
  - Sequence and Modifier wrapping ops adapted to value semantics (linear threading of all qubits including controls).

    2.2. Key Design Shifts

- Controls, inverse/adjoint, and pow(r) are modeled as explicit ops that wrap gates or sequences. Base gates have no control operands or modifier attributes.
- Sequences of unitary gates are first-class via a region-based op that only allows UnitaryOpInterface ops (including wrapped sequences). Regions use implicit terminators; explicit yield is not required in textual IR.
- Query surface is unified through a single UnitaryOpInterface with dialect-agnostic semantics and idiomatic names, while exposing both inputs and outputs for value-semantics ops.
- Ergonomics are solved with a generic builder template layered with generated overloads, and parser sugar (e.g., mqtref.ccx %c0, %c1 %q; mqtref.mcx(%c0, %c1, %c2) %q).

### 2.3. Marker Traits (Common)

- Hermitian (self-inverse): marker trait for unitary gates U with U = U^†.
  - Examples to tag: I, X, Y, Z, H (optionally SWAP, CZ, DCX after validation).
  - Canonicalization use: inv(G) -> G if G has Hermitian.
- Diagonal (computational-basis diagonal): marker trait for gates whose matrix is diagonal.
  - Examples to tag: I, Z, S, Sdg, T, Tdg, Rz, P/Phase, RZZ.
  - Analysis use: commuting/phase aggregation and specialized lowering paths.
- ODS/C++ placement:
  - CommonTraits.td: `def Hermitian : NativeOpTrait<"HermitianTrait">` and `def Diagonal : NativeOpTrait<"DiagonalTrait">`.
  - CommonTraits.h: simple marker trait declarations (no verify function required).

## 3. Types (Both Dialects)

- Qubit: the only quantum type used directly by gates in both dialects.

### 3.1 Register and Memory Handling (Quantum and Classical)

We represent quantum and classical registers using MLIR-native memrefs. This integrates cleanly with MLIR's existing memory infrastructure and passes and avoids custom register ops.

- Quantum registers: memref<k x !mqtref.Qubit> or memref<k x !mqtopt.Qubit> (depending on dialect)
  - Example (mqtref):
    ```mlir
    %qreg = memref.alloc() : memref<2x!mqtref.Qubit>
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %q0 = memref.load %qreg[%i0] : memref<2x!mqtref.Qubit>
    %q1 = memref.load %qreg[%i1] : memref<2x!mqtref.Qubit>
    memref.dealloc %qreg : memref<2x!mqtref.Qubit>
    ```
  - Example (mqtopt):
    ```mlir
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %i0 = arith.constant 0 : index
    %q0_in = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q0_out = mqtopt.h %q0_in : (!mqtopt.Qubit) -> !mqtopt.Qubit
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>
    ```

- Classical registers: memref<k x i1>
  - Measurement values (`i1`) can be stored and loaded using standard memref ops.
  - Example:
    ```mlir
    %creg = memref.alloc() : memref<1xi1>
    %c = mqtref.measure %q
    %i0 = arith.constant 0 : index
    memref.store %c, %creg[%i0] : memref<1xi1>
    memref.dealloc %creg : memref<1xi1>
    ```

Notes:

- Sequences and unitary ops act on qubit SSA values; bulk operations on registers (iteration, slicing) can be expressed with existing memref/affine utilities.

## 4. Base Gate Ops (Both Dialects)

4.1. Philosophy

- Keep base gates minimal and uniform.
- No control operands or modifier attributes.
- Only accept:
  - targets: fixed arity via TargetArity<N> trait
  - params: dynamic variadic plus optional static params via ParameterArity<N> trait and params_mask; same as current, but simplified

    4.2. Examples (mqtref)

- mqtref.x: (q: !mqtref.Qubit)
- mqtref.rx: (q: !mqtref.Qubit, params: (theta: f64|static))

  4.3. Examples (mqtopt)

- mqtopt.x: (%qin: !mqtopt.Qubit) -> (%qout: !mqtopt.Qubit)
- mqtopt.swap: (%qin0, %qin1) -> (%qout0, %qout1)

  4.4. UnitaryExpr

- Each base gate implements an interface method that returns a UnitaryExpr (symbolic, parameterized) and optionally a constant DenseElementsAttr when all parameters are static (no arity limit).

  4.5. User-defined (Arbitrary) Unitaries and Gate Definitions — Dedicated Ops (Per-Dialect)

- Motivation: Base ops intentionally cover only a curated set of primitive gates. To be universal, the IR must support user-defined unitaries in two complementary ways. We do this per dialect without introducing a separate lightweight "definitions" dialect.

- Per-dialect symbol ops at module scope
  - mqtref.unitary.def @Name [(params = [names])?] [(arity = i64)?]
    - Purpose: Define a unitary once for reference semantics. The definition may provide a constant matrix, a composite body region, or both.
    - Attributes:
      - params: list of named f64 parameters (e.g., ["theta", "phi"]).
      - arity?: required if no matrix is given; otherwise inferred from matrix shape 2^n x 2^n.
      - matrix?: DenseElementsAttr tensor<2^n x 2^n x complex<f64>> (no arity limit; fast 2x2/4x4 paths exist in UnitaryExpr).
      - traits?: ArrayAttr of StrAttr, e.g., ["Hermitian", "Diagonal"].
    - Region (optional): single-block region using mqtref.seq and unitary-capable mqtref ops. Implicit terminator.

  - mqtopt.unitary.def @Name [(params = [names])?] [(arity = i64)?]
    - Purpose: Define a unitary once for value semantics. Same attributes as above.
    - Region (optional): single-block region using mqtopt.seq and unitary-capable mqtopt ops. Must thread qubits linearly (including controls if used). Implicit terminator.

- Invocation from dialects: unitary.apply
  - mqtref.unitary.apply @Name(%q0, %q1, ...) (paramAttrs?)
    - Operands: target qubits
    - Attributes: callee = FlatSymbolRefAttr to mqtref.unitary.def; static parameters preferred (fold constants).
    - Results: none (reference semantics)
  - mqtopt.unitary.apply @Name(%qin0, %qin1, ...) (paramAttrs?) -> (%qout0, %qout1, ...)
    - Attributes: callee = FlatSymbolRefAttr to mqtopt.unitary.def; static parameters preferred.
    - Results: updated targets (and controls if wrapped by a modifier)
  - Verifier: resolve symbol; check arity/parameter counts; for body-defined unitaries, verify body well-formedness (unitary-only, linear threading in mqtopt).

- UnitaryExpr mapping
  - For matrix-defined gates, getUnitaryExpr() materializes from the DenseElementsAttr (fast 2x2/4x4 paths plus general fallback).
  - For composite-defined gates, getUnitaryExpr() is the product of child expressions in program order. Canonicalization may inline definitions at call sites when profitable.

- Examples (MLIR)
  - Fixed 1-qubit unitary via 2x2 matrix (mqtref)

    ```mlir
    // Module-scope definition: Pauli-X as a 2x2 matrix
    // tensor<2x2xcomplex<f64>> with rows [[0,1],[1,0]]
    mqtref.unitary.def @X attributes {
      matrix = dense<[[0.0+0.0i, 1.0+0.0i], [1.0+0.0i, 0.0+0.0i]]> : tensor<2x2xcomplex<f64>>,
      traits = ["Hermitian"]
    }

    // Use the definition in mqtref; modifiers can wrap it
    mqtref.unitary.apply @X(%q)
    mqtref.ctrl(%c) { mqtref.unitary.apply @X(%q) }
    ```

  - Parameterized 2-qubit composite unitary (CRZ) using a body region (mqtopt)

    ```mlir
    mqtopt.unitary.def @CRZ(params = ["theta"], arity = 2) {
      ^entry(%q0: !mqtopt.Qubit, %q1: !mqtopt.Qubit, %theta: f64):
        mqtopt.seq {
          %q1_1 = mqtopt.rz(%q1) (%theta/2.0)
          %q0_1, %q1_2 = mqtopt.cnot %q0, %q1_1 : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit)
          %q1_3 = mqtopt.rz(%q1_2) (-%theta/2.0)
          %q0_2, %q1_4 = mqtopt.cnot %q0_1, %q1_3 : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit)
        }
    }

    // Instantiate from mqtopt (value semantics)
    %q0_1, %q1_1 = mqtopt.unitary.apply @CRZ(%q0, %q1) (3.14159)
    // Wrap with modifiers
    %q0_2, %q1_2 = mqtopt.ctrl(%c) { mqtopt.unitary.apply @CRZ(%q0_1, %q1_1) (0.25) }
    ```

- Notes
  - We do not introduce a separate lightweight definitions dialect. Unitary definitions are owned by the respective dialects via unitary.def and referenced by unitary.apply.
  - Base gate ops remain per dialect (mqtref and mqtopt). Duplication is minimized by shared traits/interfaces and TableGen catalogs, plus generated builders/sugar.
  - Preliminary matrices for basis gates exist in src/dd/GateMatrixDefinitions.cpp and can seed unitary.def for standard gates during migration.

## 5. Modifiers

We introduce ops that wrap a region containing a single unitary op or a sequence op. Modifiers lazily defer; the effective unitary is composed only when queried or canonically simplified.

5.1. inv (adjoint)

- Op: mqtref.inv, mqtopt.inv
- Region: 1-region, single-block, must contain a single UnitaryOpInterface op (gate or sequence). Implicit terminator.
- Semantics: adjoint(unitary(child)).
- Canonicalization (updated):
  - inv(inv(X)) -> X
  - inv(pow(X, k)) -> pow(inv(X), k)
  - inv(ctrl(X)) -> ctrl(inv(X))
  - inv(baseGate) -> baseGateInverse with adjusted parameters

    5.2. ctrl / negctrl

- Op: mqtref.ctrl, mqtref.negctrl, mqtopt.ctrl, mqtopt.negctrl
- Operands: variadic controls of qubit type; uniqueness enforced by verifier.
- Region: as above.
- Canonicalization:
  - Merge adjacent ctrl and merge adjacent negctrl
  - Do not deduplicate or reorder controls; verifier rejects duplicates
  - Remove empty control lists
- Semantics:
  - mqtref: read-only references to controls
  - mqtopt: controls are linear values; modifiers and sequences must consume and produce new SSA values for controls. They are not simply forwarded unchanged; linear threading is enforced by types and verifiers.

    5.3. pow(r)

- Op: mqtref.pow, mqtopt.pow
- Exponent: f64 attribute or value (prefer static attribute). No parameter masks for pow.
- Canonicalization (updated):
  - pow(X, 1) -> X
  - pow(X, 0) -> identity of proper size
  - pow(pow(X, a), b) -> pow(X, a\*b)
  - pow(X, -k) -> pow(inv(X), k)

    5.4. Modifier Ordering (normal form)

- Controls must be the outermost modifiers; inverse must be the innermost. We use the normalized order:
  - innermost: inv
  - then: pow
  - outermost: ctrl/negctrl (ctrl kinds grouped and internally ordered)
- Rewriters enforce this order to avoid oscillations.

  5.5. Builder/Printer Sugar

- Single-controlled sugar for all basis gates: prefix the mnemonic with a single `c`.
  - Examples: `cx`, `cz`, `crx`, `cry`, `crz`, `cp`, `cu`, `cu2`, `cs`, `ct`, `csx`, `cswap`, …
  - Expansion: `c<g> %c, <targets>` expands to `ctrl(%c) { <g> <targets> }`
- Double-controlled sugar for common gates: `ccx` (Toffoli), `ccz`.
  - Expansion: `ccx %c0, %c1 %t` expands to `ctrl(%c0, %c1) { x %t }`
- Multi-controlled sugar: `mcx`, `mcz`, and `mcp(theta)` with arbitrarily many controls.
  - Examples:
    - `mcx(%c0, %c1, %c2) %t` expands to `ctrl(%c0, %c1, %c2) { x %t }`
    - `mcp(1.234)(%c0, %c1) %t` expands to `ctrl(%c0, %c1) { p(1.234) %t }`
- Additional compact forms remain supported:
  - `mqtref.ctrl(%c0, %c1) { mqtref.x %q }`
  - `mqtref.ctrl(%c) mqtref.inv mqtref.rx(%q) (3.14159)`

## 6. Sequence Op

6.1. Op: mqtref.seq, mqtopt.seq

- Region: 1 region, single block, contains only UnitaryOpInterface ops (including modifiers and nested seqs). Implicit terminator; explicit yield is not required in textual form.
- Traits: IsolatedFromAbove, SingleBlockImplicitTerminator<seq.yield> (terminator implicit in parser/printer), and inlining support.
- Canonicalization: inline nested seq; remove empty seq; fuse adjacent seqs; hoist/collapse modifiers across seqs when safe (e.g., inv(seq{...}) -> seq{ inv(...) reverse order }).

  6.2. Semantics

- mqtref: no results; acts on qubit references; relies on side-effect modeling as today (or NoMemoryEffect for unitary-only if desired).
- mqtopt: sequences consume and produce updated target qubits and also consume and produce controls (linear threading). Even if controls are unaffected, fresh SSA results are produced.

  6.3. Interface for seq

- The notion of "controls vs targets" is not meaningful for seq; instead expose:
  - getAllOperandQubits(): ordered list of operand qubits the sequence acts on
  - getAllResultQubits(): ordered list of result qubits the sequence produces (mqtopt only)
  - getUnitaryExpr(): the product of child UnitaryExpr in program order (M_n … M_2 M_1)

## 7. Unified UnitaryOpInterface (Common)

Given the different semantics between mqtref (reference) and mqtopt (value), the interface must be uniform yet expressive.

Idiomatic method set:

- Identification and meta
  - getIdentifier(): StringRef (op name without dialect)
  - getNumTargets(): unsigned
  - getNumControls(): unsigned
  - hasControls(): bool
  - isSingleQubit(), isTwoQubit(): bool
  - hasParams(), hasDynamicParams(), isOnlyStaticParams(): bool
- Operands/results (dialect-adaptive)
  - getTargetOperands(): OperandRange // always available
  - getControlOperands(bool positive): OperandRange // split pos/neg via getPos/NegControlOperands()
  - hasTargetResults(): bool // true in mqtopt
  - getTargetResults(): ResultRange // only valid if hasTargetResults()
  - getControlResults(bool positive): ResultRange // only in mqtopt; empty in mqtref
- Aggregate queries
  - getAllOperandQubits(): SmallVector<Value> // targets + controls (operands)
  - getAllResultQubits(): SmallVector<Value> // targets + controls (results, mqtopt)
- Unitary
  - getUnitaryExpr(): UnitaryExpr

Verification helpers in the interface ensure for value-semantics ops that in/out segment sizes are equal for targets and control kinds.

## 8. UnitaryExpr Support Library (Common)

A small C++ library to represent the underlying unitary for transformation/conversion passes only:

- Data structures
  - Tiny, fixed-size complex matrices: Mat2 (2x2), Mat4 (4x4) with stack storage (std::array<std::complex<double>, 4/16>) and constexpr helpers
  - Symbolic expression nodes: Const(Mat2/Mat4), Mul, Adj, Pow, Control(Pos/Neg with k controls), Param(index), Trig, Exp, Embed (for control structure)
- Operations
  - compose(a, b): matrix multiply / symbolic multiply
  - adjoint(), power(double r)
  - canMaterialize(): bool; materialize(Context\*) -> DenseElementsAttr for any arity (fast paths for 2x2 and 4x4)
- No external dependencies; rely on LLVM/MLIR support types (ArrayRef/SmallVector) only.
- Performance: optimized for 2x2 and 4x4; avoid heap allocs; inline-friendly.

## 9. Ergonomic Builders and C++ CircuitBuilder

Adopt a generic template design and layer nicer overloads on top.

9.1. Generic template (preferred)

- A dialect-scoped CircuitBuilder facade with a generic entry point:
  - build(opTag, Span<Value> targets, ParamSet params = {}, ControlSet posCtrls = {}, ControlSet negCtrls = {}) -> Results
- Dispatch uses trait arities to verify counts at compile time where possible.

  9.2. Generated overloads

- From StdOps.td.inc, emit overloads like:
  - x(Value q), rx(Value q, Value theta), rx(Value q, double theta)
  - cx(Value c, Value t), ccx(Value c0, Value c1, Value t)
  - mcx(ArrayRef<Value> ctrls, Value t), mcz(ArrayRef<Value> ctrls, Value t), mcp(ArrayRef<Value> ctrls, Value t, Value theta|double)
- Modifiers chain fluently: .ctrl({c...}).negctrl({c...}).inv().pow(k)

  9.3. Parser/Printer Sugar

- Compact forms supported and round-trip, including controlled sugar and multi-controlled variants.

Examples (MLIR):

```mlir
// Single-controlled
mqtref.crx %c, %t (1.234)

// Double-controlled Toffoli
mqtref.ccx %c0, %c1 %t

// Multi-controlled X and P(theta)
mqtref.mcx(%c0, %c1, %c2) %t
mqtref.mcp(1.234)(%c0, %c1) %t

// Nested modifiers
mqtref.ctrl(%c) mqtref.inv mqtref.rx(%q) (3.14159)
```

## 10. Verification and Canonicalization

10.1. Verifiers

- Base gates: enforce TargetArity and ParameterArity
- Modifiers:
  - inv: body contains exactly one UnitaryOpInterface op
  - ctrl/negctrl: unique controls; mqtopt also verifies in/out control/result segment sizes
  - pow: exponent valid; pow(0) emits identity sequence with correct arity
- Sequence: body ops implement UnitaryOpInterface; implicit terminator ok; mqtopt checks linear threading of all qubits (targets and controls)

  10.2. Canonicalization and Folds (updated rules)

- inv(inv(X)) -> X
- inv(pow(X, k)) -> pow(inv(X), k)
- inv(ctrl(X)) -> ctrl(inv(X))
- pow(X, 1) -> X; pow(X, 0) -> id; pow(pow(X, a), b) -> pow(X, a\*b)
- pow(X, -k) -> pow(inv(X), k)
- inv(baseGate) -> baseGateInverse (parameters adjusted); each base gate declares its inverse
- Controls outermost; inverse innermost; reorder modifiers accordingly
- ctrl()/negctrl() with empty list -> drop; merge adjacent modifiers; do not deduplicate or reorder controls
- Flatten nested sequences; remove empty sequences
- Fold static-parameter gates to constant UnitaryExpr where profitable

  10.3. Parameter Static Preference and Folding

- Prefer static attributes for base-gate parameters whenever possible. When a dynamic parameter operand is a compile-time constant (e.g., defined by arith.constant), fold it into the op's static_params DenseF64ArrayAttr and update params_mask accordingly.
- Mixed parameters: maintain params_mask to indicate which positions are static; if all parameters become static, remove params and params_mask entirely.
- Power modifier has no parameter mask.

Examples (mqtref):

```mlir
// Before: dynamic parameter
%pi2 = arith.constant 1.5707963267948966 : f64
mqtref.rx(%q) (%pi2)

// After: folded to static attribute
mqtref.rx(%q) (1.5707963267948966)
```

Examples (mqtopt):

```mlir
// Before: dynamic parameter with value semantics
affine.apply ... // some context producing %q_in : !mqtopt.Qubit
%pi4 = arith.constant 0.7853981633974483 : f64
%q_out = mqtopt.rz(%q_in) (%pi4) : (!mqtopt.Qubit) -> !mqtopt.Qubit

// After: static attribute preferred
%q_out = mqtopt.rz(%q_in) (0.7853981633974483) : (!mqtopt.Qubit) -> !mqtopt.Qubit
```

## 11. Passes and Pipelines

- NormalizationPass: reach normal form (modifier ordering, sequence flattening)
- ControlPushdownPass: transforms ctrl(seq{...}) into seq{ ctrl(...) ... } when valid; inverse supported
- AdjointPropagationPass: moves inv() across sequences by reversing order and adjointing gates
- ParamConstFoldPass: constant folds parameterized gates when params static
- Optional: PowerDecompositionPass: decomposes pow(r) into native gates if backend constraints require it

## 12. Conversions Between Dialects

- mqtref -> mqtopt: map base gates and modifiers; sequences become mqtopt.seq with implicit terminator; controls and targets are threaded linearly and produce fresh SSA values (including controls)
- mqtopt -> mqtref: erase value results; controls become operands only; maintain sequence/modifier structure; use inlining/materialization where required

## 13. Testing Strategy (TDD-first)

- Philosophy
  - Adopt test-driven development: write the failing LIT/unit tests first for every op, parser/printer, verifier, canonicalization, pass, and conversion; then implement the minimal code to pass them; finally refactor with tests green.
  - Prefer readable MLIR FileCheck tests for IR shape, canonicalization normal forms, and parser/printer round-trips; supplement with targeted C++ unit tests for interfaces and libraries.

- LIT: ODS/Parser/Printer and Verifiers
  - Round-trip tests for all base gates (both dialects) with static/dynamic params; ensure constant-folding favors static attrs.
  - Modifiers: nested forms, enforced ordering (controls outermost, inverse innermost), merges, and folds.
  - Sequences: inlining/flattening; implicit terminator handling (no explicit terminators required in text form).
  - Sugar: cx, cz, ccx/ccz, crx/cry/crz/cp, mcx/mcz/mcp variants; confirm desugaring to normalized IR.
  - Errors: duplicate controls (rejected by verifier), non-unitary ops inside sequences/specs, mismatched yields in value semantics, arity/parameter mismatches in unitary.apply, undefined symbols, non-power-of-two matrices.

- LIT: Unitary Definitions and Applications
  - mqt.gate.def with a fixed 2x2 matrix (e.g., X) and its application from both dialects; include a Hermitian trait check via inv canonicalization.
  - mqt.gate.def with parameters and composite spec region (e.g., CRZ); test both mqtref.unitary.apply and mqtopt.unitary.apply; wrap with modifiers and confirm canonical forms.
  - Negative cases: missing matrix and body, arity mismatch between matrix and arity attribute, illegal ops in spec region, non-linear qubit use in spec when prohibited.

- Canonicalization and Folds
  - inv(pow(X,k)) -> pow(inv(X),k); pow(X,-k) -> pow(inv(X),k); inv(base) -> base-inverse; ctrl merging; removal of empty control lists.
  - Ensure modifier ordering normalization is stable and unique.

- Interface unit tests (C++)
  - UnitaryOpInterface: target/control operand/result queries across base, modifiers, sequences, and unitary.apply.
  - UnitaryExpr: adjoint, power, composition, 2x2/4x4 multiply, and dense materialization for larger arities.

- Builder API (C++)
  - Compile-time tests for the generic template and generated overloads.
  - Runtime smoke tests building small circuits, including sugar (cx, ccx, mcx/mcz/mcp) and nested modifiers.

- Pass and Conversion tests
  - Normalization, ControlPushdown, AdjointPropagation, ParamConstFold.
  - mqtref<->mqtopt conversions with modifiers/sequences and linear control threading; preserve normal form.

    13.0. Checkstring Robustness and Idiomatic MLIR Testing

To minimize false negatives from fragile FileCheck patterns and to keep tests maintainable:

- Prefer round‑trip tests: `RUN: mlir-opt %s | FileCheck %s` and, where appropriate, `RUN: mlir-opt %s | mlir-opt | FileCheck %s` to ensure printers/parsers are robust.
- Normalize before checking when the exact textual form is not the goal: add `-canonicalize -cse` to the pipeline to check semantic shape rather than incidental formatting.
- Use `-split-input-file` to host many small, focused test cases in one file.
- For negative tests, use `-verify-diagnostics` with `// expected-error`/`// expected-note` comments instead of FileCheck.
- Avoid matching SSA value numbers; anchor on op names, attributes, and structural patterns. Use `CHECK-LABEL:` for sectioning, `CHECK-SAME:` to continue lines, and `CHECK-DAG:` for order‑insensitive matches where appropriate.
- Where printer formatting may change, prefer `-mlir-print-op-generic` to stabilize tests.
- For floating‑point literals, avoid brittle exact decimal matches; either rely on canonicalization (e.g., constants folded) or match with tolerant regex patterns.
- For parser/printer sugar, add paired tests: one that checks the sugared input parses to the normalized IR, and another that checks the normalized IR prints back to the sugared form when expected.
- Add programmatic C++ unit tests to construct and verify IR:
  - Parse with `mlir::parseSourceString`, run passes with `mlir::PassManager`, and check invariants via `Operation::verify()`.
  - Build IR with `OpBuilder` and the CircuitBuilder API, print to string, and compare with FileCheck‑style matchers or targeted substring assertions.

Example (LIT skeleton):

```mlir
// RUN: mlir-opt %s -mqtref-normalize -canonicalize | FileCheck %s

// CHECK-LABEL: func @demo
func.func @demo() {
  %q = mqtref.allocQubit
  // CHECK: mqtref.ctrl(
  mqtref.ctrl(%q) { mqtref.x %q }
  return
}
```

## 14. File/Code Changes (Planned)

14.1. Common (shared)

- New: mlir/include/mlir/Dialect/Common/IR/QuantumInterfaces.td (consolidated UnitaryOpInterface)
- New: mlir/include/mlir/Dialect/Common/IR/UnitaryExpr.h/.cpp (support library)
- Update: mlir/include/mlir/Dialect/Common/IR/CommonTraits.td/.h (add helper traits if needed)
- New: mlir/include/mlir/Dialect/Common/IR/ParserPrinterUtils.h/.cpp for modifiers/params/sequence printing and sugar
- New: mqt definitions dialect for unitary definitions
  - Headers: mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td, MQTOps.td (mqt.gate.def, mqt.gate.decl)
  - Impl: mlir/lib/Dialect/MQT/IR/MQTDialect.cpp, MQTOps.cpp (parser/printer/verifier)

    14.2. mqtref

- Update: MQTRefOps.td
  - Redefine base UnitaryOp without control operands
  - Add Modifiers: ctrl, negctrl, inv, pow (with implicit region terminators)
  - Add SequenceOp (implicit terminator)
  - Add Unitary invocation op: unitary.apply (call-site implementing UnitaryOpInterface)
  - Use dedicated definition ops: mqt.gate.def (and optional mqt.gate.decl) in the new mqt definitions dialect; no func.func attributes
  - Keep Resource/Reset/Measure as-is
- Update: MQTRefInterfaces.td -> adopt Common UnitaryOpInterface
- New: MQTRefCircuitBuilder.h/.cpp (generated from StdOps.td.inc)

  14.3. mqtopt

- Update: MQTOptOps.td
  - Redefine base UnitaryOp without control operands; value results for targets
  - Add Modifiers: ctrl/negctrl (linear threading of controls), inv, pow (implicit terminators)
  - Add SequenceOp (implicit terminator) with linear threading
  - Add Unitary invocation op: unitary.apply (call-site implementing UnitaryOpInterface)
  - Use dedicated definition ops from the mqt definitions dialect: mqt.gate.def (and optional mqt.gate.decl); no func.func attributes
- Update: MQTOptInterfaces.td -> adopt Common UnitaryOpInterface; ensure queries expose both operands and results as specified
- New: MQTOptCircuitBuilder.h/.cpp (generated)

  14.4. TableGen Codegen Enhancements

- Extend StdOps.td.inc to also emit builder helpers via a new include (e.g., MQTRefCircuitBuilder.inc / MQTOptCircuitBuilder.inc) using preprocessor macros
- Prefer the generic template entry-point and generate thin overloads on top

  14.5. Passes

- New pass registrations and implementations in mlir/lib/.../Transforms for normalization, propagation, folding

  14.6. Conversions

- Update/refactor existing conversions to respect new modifier/sequence ops

  14.7. Documentation

- Update docs/mlir/Conversions.md and write new docs/mlir/Dialect.md for user-facing syntax and builder API
- Docstring guideline: every op and modifier must include fenced MLIR examples demonstrating sugar and canonicalized forms

## 15. Migration and Deprecation

- No backward compatibility required; remove controls from base gates and replace with modifiers.
- No temporary helper conversion will be provided; all code is expected to be rewritten and fixed accordingly (tests will be updated alongside).
- QubitRegisters have been replaced with memrefs; this plan reflects the current state.

## 16. Risks and Mitigations

- Interface uniformity: Ensure UnitaryOpInterface exposes both operand and (optional) result views; provide hasTargetResults() and control result accessors to avoid dialect-specific branching in clients.
- Performance of UnitaryExpr: Keep operations cheap; favor small fixed-size matrices; avoid materialization unless necessary; support dense materialization for any arity (with fast paths for 2x2 and 4x4).
- Modifier/sequence ordering: Define strict normal form (controls outermost, inverse innermost); add extensive LIT to prevent rewrite loops.
- Linear controls in mqtopt: Provide utilities for segment attrs and builders that always produce fresh SSA for controls.

## 17. Milestones and Work Breakdown

M1 — Foundations (Common) [~1–2 weeks]

- Create QuantumInterfaces.td with UnitaryOpInterface ✓
- Implement UnitaryExpr skeleton (adjoint, power, compose, 2x2/4x4 multiply) ✓
- Parser/Printer utils for params, modifiers, sugar ✓

M2 — mqtref Base + Modifiers + Sequence [~2 weeks]

- Redefine base UnitaryOp (no controls) ✓
- Implement ctrl/negctrl/inv/pow with verifiers and canonicalization ✓
- Implement seq with inliner and canonicalization; implicit terminator ✓
- Implement unitary.apply (instantiation) ✓
- Use dedicated mqt.gate.def definitions (and optional mqt.gate.decl); add verifiers/utilities ✓
- Update std gate defs and assembly formats ✓
- LIT tests: ops, parsers, canonicalization, mqt.gate.def and unitary.apply ✓

M3 — mqtopt Mirror [~2 weeks]

- Redefine base gate ops with value outputs ✓
- Implement modifiers with linear control threading; sequence with implicit terminator ✓
- Implement unitary.apply (instantiation) ✓
- Use mqt.gate.def for definitions (and optional mqt.gate.decl); add verifiers/utilities ✓
- Update interface and verifiers ✓
- LIT tests mirroring mqtref, including mqt.gate.def and unitary.apply ✓

M4 — Builders and Ergonomics [~1 week]

- Generate CircuitBuilder helper API with generic template + overloads ✓
- Parser/printer sugar for compact modifier nesting and cx/cz/ccx/ccz/mcx/mcz/mcp ✓
- C++ smoke tests ✓

M5 — Passes and Conversions [~1–2 weeks]

- Normalization, ControlPushdown, AdjointPropagation, ParamConstFold ✓
- Update mqtref<->mqtopt conversions for modifiers/sequences with linear controls ✓
- Tests ✓

M6 — Documentation and Polishing [~1 week]

- Update docs and examples; ensure all docstrings contain fenced MLIR examples ✓
- Final test stabilization ✓

Note: ✓ indicates planned completion within this revamp; actual sequencing will be tracked in the repository issue tracker.

## 18. Acceptance Criteria

- Both dialects compile and pass LIT tests covering base gates, modifiers, sequences, mqt.gate.def definitions, unitary.apply, and printers (including sugar like cx, ccx, mcx, mcp).
- Unified UnitaryOpInterface available, used by all unitary-capable ops, and provides dialect-adaptive input/output queries.
- Support for arbitrary unitary definitions via dedicated ops: mqt.gate.def (matrix-defined and composite-defined) and unitary.apply in both dialects; modifiers can wrap applications.
- CircuitBuilder API available, with generic template and overloads for core gates; sugar for single- and multi-controlled gates.
- Canonicalization suite ensures normalized IR: controls outermost; inverse innermost; pow/inv rules as specified; linear controls in mqtopt.
- Conversions between mqtref and mqtopt handle modifiers/sequences and linear threading of controls.

## 19. Illustrative Examples

19.1. mqtref (IR)

```mlir
// prepare qubits
%q0 = mqtref.allocQubit
%q1 = mqtref.allocQubit
%c0 = mqtref.allocQubit
%c1 = mqtref.allocQubit
%c2 = mqtref.allocQubit

// Single-controlled
mqtref.crz %c0, %q0 (0.25)

// Double-controlled Toffoli
mqtref.ccx %c0, %c1 %q0

// Multi-controlled
mqtref.mcx(%c0, %c1, %c2) %q1
mqtref.mcp(1.234)(%c0, %c1) %q0

// Nested modifiers & canonicalization
mqtref.inv mqtref.ctrl(%c0) { mqtref.rx(%q0) (3.14159) }
// -> mqtref.ctrl(%c0) { mqtref.inv mqtref.rx(%q0) (3.14159) }
```

19.2. mqtopt (IR)

```mlir
%q0 = mqtopt.allocQubit
%q1 = mqtopt.allocQubit
%c  = mqtopt.allocQubit

// Sequence returns updated targets; controls are threaded linearly
mqtopt.seq {
  %q0_1 = mqtopt.h %q0
  // ctrl threads control linearly: consumes %c, yields %c_1
  %q0_2, %c_1 = mqtopt.ctrl(%c) { mqtopt.x %q0_1 }
}
```

19.3. Canonicalization

```mlir
// inv(pow(X, k)) -> pow(inv(X), k)
mqtref.inv mqtref.pow(3) { mqtref.x %q }  // -> mqtref.pow(3) { mqtref.inv mqtref.x %q }

// pow(X, -k) -> pow(inv(X), k)
mqtref.pow(-2) { mqtref.h %q }  // -> mqtref.pow(2) { mqtref.inv mqtref.h %q }

// inv(baseGate) -> baseGateInverse
mqtref.inv mqtref.s %q  // -> mqtref.sdg %q

// inv(ctrl(X)) -> ctrl(inv(X))
mqtref.inv mqtref.ctrl(%c) { mqtref.rz(%q) (0.5) }
// -> mqtref.ctrl(%c) { mqtref.inv mqtref.rz(%q) (0.5) }
```

19.4. C++ Builder

```cpp
using namespace mqt::ir::ref;
CircuitBuilder qb(b, loc);
auto q0 = qb.allocQubit();
auto q1 = qb.allocQubit();
auto c0 = qb.allocQubit();
auto c1 = qb.allocQubit();

qb.cx(c0, q0);                        // CNOT sugar
qb.ccx(c0, c1, q0);                   // Toffoli sugar
qb.mcx({c0, c1}, q1);                 // Multi-controlled X
qb.mcp({c0}, q0, M_PI/4.0);           // Multi-controlled phase
qb.rx(q0, M_PI/2).inv();              // R_x(pi/2)^-1 (inverse as innermost in normal form)
qb.seq([&]{ qb.h(q0); qb.rz(q0, t); });
```

## 20. Documentation Guidelines

- Every op (base, modifier, seq) must include fenced MLIR examples in the ODS docstrings, demonstrating both verbose and sugar forms and at least one canonicalization result.
- Use idiomatic names in MLIR, C++, and QC contexts: UnitaryOpInterface, getTargetOperands, getTargetResults, getPosControlOperands, getNegControlOperands, getUnitaryExpr, getAllOperandQubits, getAllResultQubits, etc.

## 21. Next Steps

- Proceed to implementation per milestones now that the revisions are confirmed.
