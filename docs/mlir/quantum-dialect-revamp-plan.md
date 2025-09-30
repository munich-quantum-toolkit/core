# Quantum Dialect Revamp RFC

## 1. Overview and Goals

This RFC proposes a comprehensive revamp of the quantum MLIR dialect(s) to unify unitary representations, improve expressiveness, and enable robust transformations.

Goals:

- Introduce a single abstraction: **UnitaryExpr** (symbolic unitary expression) replacing prior matrix vs expression split.
- Provide a coherent interface for all operations that apply or produce a unitary (base gates, user-defined gates, modifier-wrapped constructs, sequences).
- Support both reference semantics (in-place: `mqtref`) and value semantics (SSA threading: `mqtopt`).
- Add inversion, powering, negative and positive multi-controls, custom gate definitions (matrix & composite), and composition.
- Unify parameter handling (static + dynamic) with consistent ordering and interface queries.
- Embed canonicalization rules directly at each operation definition.
- Establish a normalized modifier nesting order: `negctrl → ctrl → pow → inv`.
- Enable static matrix extraction where possible and symbolic composition otherwise.
- Prepare foundations for advanced transformations without speculative overreach.

## 2. Current State and Limitations

Current issues the revamp addresses:

- Only a rudimentary control modifier exists; missing negative controls, power, inversion.
- No unified interface for extracting/composing unitaries—leading to ad hoc logic.
- Matrix vs expression forms diverge; lost optimization opportunities.
- No user-defined gate (matrix/composite) constructs.
- Inconsistent parameter model and missing static/dynamic integration.
- Lack of dual semantics for optimization vs generation workflows.
- Absent canonicalization strategy for modifier order, parameter folding, gate specialization.

## 3. Dialect Structure and Categories

Two parallel dialects:

- `mqtref`: Reference semantics (in-place mutation of qubits; no new results).
- `mqtopt`: Value semantics (operations consume qubits and yield new qubit results).

Categories:

1. Resource Operations
2. Measurement and Reset
3. UnitaryExpr Concept (applies everywhere for unitaries)

### 3.1 Resource Operations

Purpose: Manage qubit lifetime and references.

Examples (reference semantics):

```
%q = mqtref.alloc : mqtref.Qubit
mqtref.dealloc %q : mqtref.Qubit
%q_fixed = mqtref.static_qubit @q0 : mqtref.Qubit
```

Value semantics:

```
%q0 = mqtopt.alloc : mqtopt.Qubit
mqtopt.dealloc %q0 : mqtopt.Qubit
%q_hw = mqtopt.static_qubit @q7 : mqtopt.Qubit
```

Canonicalization (patterns / folds):

- Remove unused `alloc` (DCE).
- Elide `dealloc` proven by lifetime analysis.
- Merge duplicate `static_qubit` references if semantics allow.

### 3.2 Measurement and Reset

Non-unitary (do not implement Unitary interface).

Reference:

```
%c = mqtref.measure %q : mqtref.Qubit -> i1
mqtref.reset %q : mqtref.Qubit
```

Value:

```
%c = mqtopt.measure %qin : mqtopt.Qubit -> i1
%qout = mqtopt.reset %qin : mqtopt.Qubit
```

Canonicalization:

- `reset` immediately after `alloc` → remove `reset`.
- Consecutive `reset` on same qubit (reference semantics) → single instance.

### 3.3 UnitaryExpr Concept

`UnitaryExpr` is a conceptual abstraction representing a (possibly symbolic) unitary over n targets with optional parameters and controls.

- Static if all parameters are static and analytic matrix is available.
- Symbolic otherwise (composition, parameterized nodes, modifiers).
- Provides inversion, powering, and control extension without immediate matrix materialization.

## 4. Unified Unitary Interface Design

All unitary-applying operations implement a common interface (applies to base gates, modifiers, sequences, and user-defined applies).

Interface methods (conceptual API):

- `getNumTargets() -> unsigned`
- `getNumPosControls() -> unsigned`
- `getNumNegControls() -> unsigned`
- `getNumParams() -> unsigned`
- `getParameter(i) -> ParameterDescriptor`
  - `ParameterDescriptor`: `isStatic()`, `getConstantValue()?`, `getValueOperand()`
- `getInput(i)` / `getOutput(i)` (value semantics distinct; reference semantics output = input)
- `mapOutputToInput(i) -> i` (pure unitaries)
- `hasStaticUnitary() -> bool`
- `getOrBuildUnitaryExpr(builder) -> UnitaryExpr`
- `tryGetStaticMatrix() -> Optional<Attribute>` (2D tensor with shape (2^n, 2^n) and element type `complex<f64>`; written concretely for fixed n as e.g. `tensor<4x4xcomplex<f64>>`)
- `isInverted() -> bool`
- `getPower() -> Optional<RationalOrFloat>`
- `withAddedControls(pos, neg) -> UnitaryExpr`
- `composeRight(other) -> UnitaryExpr`
- `getPrincipalLog() -> Optional<Symbolic>`

Identification & Descriptor Tuple:
`(baseSymbol, orderedParams, posControls, negControls, powerExponent, invertedFlag)` allows canonical equality tests.

Parameter Model:

- Parameters appear in parentheses immediately after mnemonic.
- Mixed static (attributes) and dynamic (operands) preserve original order.
- Enumeration returns flattened ordered list; inspect each for static/dynamic.

Static Matrix Extraction:

- Provided if gate analytic and all parameters static, or for matrix-defined user gates.
- For sequences/composites of static subunits under size threshold, compose matrices.

Inversion & Power Interaction:

- `inv` introduces `invertedFlag` (final canonical position).
- `pow` stores exponent; negative exponent canonicalized to `inv(pow(+exp))` then reordered.

Control Extension:

- `ctrl` / `negctrl` wrappers extend control sets; interface aggregates flattened sets.

## 5. Base Gate Operations

### 5.1 Philosophy

- Named base gates define analytic unitaries with fixed target arity and parameter arity traits.
- Provide static matrix when parameters static; symbolic otherwise.
- Avoid embedding modifier semantics directly—wrappers handle extension.

### 5.2 Gate List (Single-Qubit)

No-parameter: `x, y, z, h, s, sdg, t, tdg, id`
Parameterized: `rx(%theta), ry(%theta), rz(%theta), phase(%lambda), u(%theta, %phi, %lambda)`

Multi-qubit illustrative: `rzz(%theta)` (two targets), not using `cx` (introduced only via sugar as controlled `x`).

#### 5.2.1 Base Gate Specification Template

For every named base gate op G:

- Purpose: Apply the analytic unitary for gate G to its target qubit(s).
- Signature (Reference): `mqtref.G(param_list?) %q[,...] : <param types..., qubit types>` (no results)
- Signature (Value): `%out_targets = mqtopt.G(param_list?) %in_targets : (<param types..., qubit types>) -> <qubit types>`
- Assembly Format: `G(<params?>) <targets>`; params in parentheses; qubits as trailing operands.
- Builder Variants:
  - `build(builder, loc, resultTypes, paramOperands, qubitOperands)` (value)
  - `build(builder, loc, qubitOperands, paramAttrs)` (reference)
  - Convenience: static param overloads generate attribute parameters.
- Interface Implementation Notes:
  - `getNumTargets()` fixed by trait.
  - Parameters enumerated in declared order; static vs dynamic via attribute vs operand.
  - `hasStaticUnitary()` true iff all parameters static.
  - `mapOutputToInput(i) = i`.
- Canonicalization Rules: See 5.6 plus gate-specific (e.g., identity elimination, specialization).
- Examples (Static): `mqtref.rz(3.14159) %q`; `%q2 = mqtopt.rx(0.785398) %q1`.
- Examples (Dynamic): `%q2 = mqtopt.rx(%theta) %q1`; `mqtref.u(%t,%p,%l) %q`.
- Conversion (ref↔value): Reference variant lowers to value variant with SSA replacement; reverse drops result.

#### 5.2.2 Example: rx Gate

- Purpose: Single-qubit rotation about X by angle θ.
- Signatures:
  - Ref: `mqtref.rx(%theta) %q : f64, mqtref.Qubit`
  - Value: `%q_out = mqtopt.rx(%theta) %q_in : (f64, mqtopt.Qubit) -> mqtopt.Qubit`
- Static Example: `%q_out = mqtopt.rx(1.57079632679) %q_in`
- Dynamic Example: `%q_out = mqtopt.rx(%theta) %q_in`
- Canonicalization: `rx(0) → id`; two consecutive `rx(a); rx(b)` NOT folded (axis change would require Baker-Campbell-Hausdorff? skip); `inv rx(θ)` handled by modifier → `rx(-θ)`.
- Static Matrix Available: Yes if θ constant.

#### 5.2.3 Example: rzz Gate

- Purpose: Two-qubit entangling gate `exp(-i θ/2 Z⊗Z)`.
- Signatures:
  - Ref: `mqtref.rzz(%theta) %q0, %q1 : f64, mqtref.Qubit, mqtref.Qubit`
  - Value: `%q0_out, %q1_out = mqtopt.rzz(%theta) %q0_in, %q1_in : (f64, mqtopt.Qubit, mqtopt.Qubit) -> (mqtopt.Qubit, mqtopt.Qubit)`
- Static Example: `%a1, %b1 = mqtopt.rzz(3.14159) %a0, %b0`
- Dynamic Example: `%a1, %b1 = mqtopt.rzz(%theta) %a0, %b0`
- Canonicalization: `rzz(0) → id`; `inv rzz(θ) → rzz(-θ)`.
- Static Matrix Available: Yes if θ constant.

### 6. Modifier Operations

### 6.1 Overview

Modifiers wrap unitaries, extending functionality or altering semantics.

- Semantics-preserving (e.g., `ctrl`, `negctrl`): canonicalized order, flattened.
- Transformative (e.g., `pow`, `inv`): applied last, may alter static matrix extraction.

### 6.2 negctrl

Purpose: Add negative controls.

Operation Specification:

- Purpose: Wrap a unitary adding negative (0-state) control qubits.
- Signatures:
  - Ref: `mqtref.negctrl %negControls { <unitary-body> }`
  - Value: `%res_targets = mqtopt.negctrl %negControls { <yielded unitary> } -> <qubit types>`
- Assembly: `negctrl <ctrl-list> { ... }`.
- Builder Variants:
  - `build(builder, loc, resultTypes, negControlOperands, bodyBuilderFn)` (value)
  - Reference variant omits results.
- Interface Notes: Aggregates controls into `getNumNegControls()`; targets delegated to child.
- Canonicalization: Flatten nested, remove empty, reorder relative to other modifiers to canonical chain `negctrl → ctrl → pow → inv`.
- Examples:
  - Ref: `mqtref.negctrl %n0 { mqtref.h %t }`
  - Value: `%t_out = mqtopt.negctrl %n0 { %t1 = mqtopt.rx(%theta) %t_in } -> mqtopt.Qubit`
- Conversion: Region body value results threaded / dropped analogously to other wrappers.

### 6.3 ctrl

Operation Specification:

- Purpose: Add positive (1-state) controls.
- Signatures:
  - Ref: `mqtref.ctrl %posControls { <unitary-body> }`
  - Value: `%res_targets = mqtopt.ctrl %posControls { <yielded unitary> } -> <qubit types>`
- Builder: Similar to `negctrl` with positive control list.
- Interface: `getNumPosControls()` sums flattened list.
- Canonicalization: Merge nested, remove empty, optionally distribute over `seq`, enforce order after `negctrl`.
- Examples:
  - Ref: `mqtref.ctrl %c { mqtref.rzz(%φ) %q0, %q1 }`
  - Value: `%t_out = mqtopt.ctrl %c { %t1 = mqtopt.rz(%φ) %t_in } -> mqtopt.Qubit`
- Conversion: As for `negctrl`.

### 6.4 pow

Operation Specification:

- Purpose: Exponentiation of a unitary body.
- Signatures:
  - Ref: `mqtref.pow(expAttrOrOperand) { <unitary-body> }`
  - Value: `%res_targets = mqtopt.pow(expAttrOrOperand) { <yielded unitary> } -> <qubit types>`
- Assembly: `pow(<int|float|%val>) { ... }`.
- Builder Variants: integer attribute exponent; float attribute; dynamic f64 operand.
- Interface: `getPower()` returns rational/float wrapper; static detection when attribute.
- Canonicalization: Negative -> `inv(pow(abs))`; combine nested powers; remove exponent 1; exponent 0 -> identity passthrough; reorder with other modifiers.
- Examples:
  - `%q2 = mqtopt.pow(2) { %q1 = mqtopt.rx(%theta) %q0 }`
  - `%q2 = mqtopt.pow(%k) { %q1 = mqtopt.rz(%φ) %q0 }`
- Conversion: Same region adaptation logic.

### 6.5 inv

Operation Specification:

- Purpose: Adjoint of unitary body.
- Signatures:
  - Ref: `mqtref.inv { <unitary-body> }`
  - Value: `%res_targets = mqtopt.inv { <yielded unitary> } -> <qubit types>`
- Builder: Provide body builder lambda.
- Interface: `isInverted()` true; nested inversion removed in canonicalization.
- Canonicalization: Double inversion removal; self-adjoint detection; distribute over `pow` forms (placing `inv` innermost after ordering); axis negation for parameterized rotations.
- Examples: `%t_out = mqtopt.inv { %t1 = mqtopt.u(%theta,%phi,%lambda) %t_in }`
- Conversion: Same as other wrappers.

### 6.6 Nested Example

Original value form (non-canonical):

```
%out = mqtopt.inv { %a = mqtopt.ctrl %c { %b = mqtopt.negctrl %n { %g = mqtopt.rx(%theta) %in } } } -> mqtopt.Qubit
```

Canonical extraction: negctrl(%n), ctrl(%c), inv.
Reordered canonical:

```
%out = mqtopt.negctrl %n {
  %t1 = mqtopt.ctrl %c {
    %t2 = mqtopt.inv { %t3 = mqtopt.rx(%theta) %in } -> mqtopt.Qubit
  } -> mqtopt.Qubit
} -> mqtopt.Qubit
```

After folding `inv rx(%theta)` → `rx(-%theta)`:

```
%out = mqtopt.negctrl %n {
  %t1 = mqtopt.ctrl %c { %t2 = mqtopt.rx(-%theta) %in } -> mqtopt.Qubit
} -> mqtopt.Qubit
```

Reference nested example (explicit):

```
mqtref.negctrl %n {
  mqtref.ctrl %c {
    mqtref.inv { mqtref.rx(%theta) %q }
  }
}
```

After folding: `mqtref.negctrl %n { mqtref.ctrl %c { mqtref.rx(-%theta) %q } }`

## 7. Sequence Operation (`seq`)

Purpose: Ordered composition of unitary operations over region block arguments.

Operation Specification:

- Purpose: Represent composite product U = U_n … U_2 U_1 in region order.
- Signatures:
  - Ref: `mqtref.seq (%args: mqtref.Qubit, ...) { <unitary-ops> mqtref.seq.yield }`
  - Value: `%results = mqtopt.seq (%args: mqtopt.Qubit, ...) -> (mqtopt.Qubit, ...) { <ops> mqtopt.seq.yield %newArgs }`
- Assembly: `seq (arg_list) -> (result_types)? { ... }`
- Builders: Provide argument list + body builder capturing yields; value builder generates result types from argument types.
- Interface: Targets = block argument count; parameters aggregated from children (none directly); `hasStaticUnitary()` if all child ops static and compose cost acceptable.
- Canonicalization: Flatten nested, remove empty, inline single op, distribute controls from wrappers if beneficial, inversion reversal & child inversion patterns.
- Examples:

```
%q0_out, %q1_out = mqtopt.seq (%q0_in: mqtopt.Qubit, %q1_in: mqtopt.Qubit)
  -> (mqtopt.Qubit, mqtopt.Qubit) {
  %a0 = mqtopt.h %q0_in
  %b0, %b1 = mqtopt.rzz(%θ) %a0, %q1_in
  %c0 = mqtopt.rz(%φ) %b0
  mqtopt.seq.yield %c0, %b1
}
```

- Conversion: Reference ↔ value via region argument threading and yields.

## 8. User Defined Gates & Matrix / Composite Definitions

### 8.1 Matrix Gate Definitions

Define symbol with matrix attribute (dimension 2^n × 2^n):

```
mqt.gatedef.matrix @myPhase(%lambda: f64) targets(1)
  attr_matrix = #mqt.matrix<2x2>( ... symbolic in %lambda ... )
```

Matrix may embed symbolic expressions referencing parameters (internal representation detail).

### 8.2 Composite Gate Definitions

Sequence-based body yields outputs (value semantics):

```
mqt.gatedef.composite @entang(%theta: f64)
  ( %a: mqtopt.Qubit, %b: mqtopt.Qubit ) -> (mqtopt.Qubit, mqtopt.Qubit) {
  %a1 = mqtopt.h %a
  %a2, %b1 = mqtopt.rzz(%theta) %a1, %b
  mqtopt.seq.yield %a2, %b1
}
```

Reference semantics variant has same region arguments but no gate results.

### 8.3 Unified Apply Operation

Value:

```
%a_out, %b_out = mqtopt.apply @entang(%theta) %a_in, %b_in
%q1 = mqtopt.apply @myPhase(%lambda) %q0
```

Reference:

```
mqtref.apply @entang(%theta) %a, %b
mqtref.apply @myPhase(3.14159) %q
```

Operation Specification:

- Purpose: Apply user-defined matrix or composite gate symbol.
- Signatures:
  - Ref: `mqtref.apply @symbol(param_list?) %targets`
  - Value: `%results = mqtopt.apply @symbol(param_list?) %inputs`
- Assembly: `apply @name(<params?>) <targets>`
- Builder Variants:
  - For matrix gate: auto infer target count from definition.
  - For composite: verify arity against definition region signature.
  - Parameter list builder with static attribute injection.
- Interface Notes:
  - `getNumTargets()` from definition signature.
  - Parameters enumerated exactly as in definition.
  - `hasStaticUnitary()` true for matrix gate; composite conditional.
- Canonicalization: Identity matrix removal; trivial single-op composite inlining; repeated static collapses per algebraic rules.
- Examples:
  - Static parameter: `%q1 = mqtopt.apply @myPhase(3.14159) %q0`
  - Dynamic parameter: `%q1 = mqtopt.apply @myPhase(%lambda) %q0`
- Conversion: Reference ↔ value semantics by adding/removing results and adjusting uses.

## 9. Parser & Builder Sugar

### 9.1 Sugar Expansions

- `cx %c, %t` → `mqtopt.ctrl %c { %t_out = mqtopt.x %t }` (value)
- `cz %c, %t` → `ctrl %c { z %t }`
- `ccx %c0, %c1, %t` → `ctrl %c0, %c1 { x %t }`
  Parser lowers sugar to canonical IR; printer may re-sugar canonical forms that match patterns.

### 9.2 Fluent Builder API (Conceptual)

Examples:

```
withControls({c1,c2}).gate("x").on(t)
withNegControls({n}).gate("rz").param(phi).on(t)
withPow(3).gate("h").on(q)
withInv().gate("u").params(a,b,c).on(q)
sequence({}) RAII style for `seq`
defineMatrixGate("myPhase").params({lambda}).targets(1).matrix(attr)
defineCompositeGate("entang").params({theta}).targets(2).body([]{...})
apply("entang").params(theta).on(q0,q1)
```

Chaining order auto-normalized to canonical modifier order.

## 10. Testing Strategy

- Structural: verify canonical modifier order and flattening.
- Matrix correctness: `U†U = I` for static extractions.
- Interface conformance: each op's counts (targets, controls, params) correct; mapping output→input identity for pure unitaries.
- Canonicalization idempotence: run pass twice, IR stable.
- Sugar round-trip: parse sugar → canonical → print (optionally sugar) with equivalence.
- Folding tests: `rx(0)`, `pow(1){}`, `pow(0){}`, `inv(inv(U))`.
- Negative exponent normalization tests.
- Sequence inversion correctness via static matrix comparison for small systems.
- Apply inlining & identity elimination tests.
- Negative tests: arity mismatch, invalid matrix dimension, non-unitary matrix, duplicate symbol definition, parameter count mismatch.

## 11. Integrated Canonicalization Rules Summary

(Definitions live inline above; this section aggregates references.)

- Base Gates: parameter folds, identity elimination, specialization to named gates.
- `negctrl`: flatten & remove empty.
- `ctrl`: flatten, remove empty, distribute over `seq` when beneficial.
- `pow`: normalize negatives, remove trivial exponents, combine nested powers.
- `inv`: double-inversion removal, specialize to inverses/self-adjoint forms.
- Modifier Ordering: reorder to `negctrl → ctrl → pow → inv`.
- `seq`: flatten, remove empty, inline single-op.
- `apply`: inline trivial composite, fold identities, merge repeated static applies.
- Sugar: lower to canonical, optional re-sugar on print.

## 12. Conclusions and Future Work

Resolved:

- Unified `UnitaryExpr` abstraction with full modifier and composition support.
- Static + dynamic parameter integration and robust interface methods.
- Canonical nested modifier ordering and explicit rewrite rules.
- User-defined gates (matrix & composite) plus unified `apply` op.
- Dual semantics (`mqtref`, `mqtopt`) standardized.

Future Work (concise):

- Basis decomposition (e.g., KAK, ZX-assisted) using `UnitaryExpr` graphs.
- Shared control extraction & factoring.
- Symbolic algebra simplifications (parameter expression normalization).
- Hardware mapping leveraging `static_qubit` references.
