# Quantum Dialect Revamp RFC

## 1. Overview and Goals

This RFC proposes a comprehensive revamp of the quantum MLIR dialect(s) to unify unitary representations, improve expressiveness, and enable robust transformations.

Goals:

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

- Only a control modifier exists and is directly embedded in the unitary operations (interface); missing power, inversion.
- No unified interface for extracting/composing unitaries—leading to ad hoc logic.
- No way to obtain matrix representations for gates.
- No user-defined gate (matrix/composite) constructs.
- Absent canonicalization strategy for modifier order, parameter folding, gate specialization.
- Mostly FileCheck-based testing that is cumbersome and error prone to write.
- No convenient builders for programs at the moment.

## 3. Dialect Structure and Categories

Two parallel dialects:

- `mqtref`: Reference semantics (in-place mutation of qubits; no new results).
- `mqtopt`: Value semantics (operations consume qubits and yield new qubit results).

Categories:

1. Resource Operations
2. Measurement and Reset
3. UnitaryInterface Operations

### 3.1 Resource Operations

Purpose: Manage qubit lifetime and references.

Examples (reference semantics):

```
%q = mqtref.alloc : mqtref.Qubit
mqtref.dealloc %q : mqtref.Qubit
%q0 = mqtref.qubit 0 : mqtref.Qubit
```

Value semantics:

```
%q = mqtopt.alloc : mqtopt.Qubit
mqtopt.dealloc %q : mqtopt.Qubit
%q0 = mqtopt.qubit 0 : mqtopt.Qubit
```

Canonicalization (patterns / folds):

- Remove unused `alloc` (DCE).
- Elide `dealloc` proven by lifetime analysis. // TODO
- Merge duplicate `qubit` references if semantics allow. // TODO

### 3.2 Measurement and Reset

Non-unitary (do not implement Unitary interface).

Reference:

```
%c = mqtref.measure %q : mqtref.Qubit -> i1
mqtref.reset %q : mqtref.Qubit
```

Value:

```
%qout, %c = mqtopt.measure %qin : mqtopt.Qubit -> (mqtopt.Qubit, i1)
%qout = mqtopt.reset %qin : mqtopt.Qubit
```

Canonicalization:

- `reset` immediately after `alloc` → remove `reset`.
- Consecutive `reset` on same qubit (reference semantics) → single instance.

### 3.3 Unified Unitary Interface Design

All unitary-applying operations implement a common interface (applies to base gates, modifiers, sequences, and user-defined operations).

Interface methods (conceptual API):

- `getNumTargets() -> size_t`
- `getNumPosControls() -> size_t`
- `getNumNegControls() -> size_t`
- `getNumParams() -> size_t`
- `getParameter(i) -> ParameterDescriptor`
  - `ParameterDescriptor`: `isStatic()`, `getConstantValue()?`, `getValueOperand()`
- `getInput(i)` / `getOutput(i)` (value semantics distinct; reference semantics output = input)
- `mapOutputToInput(i) -> i` (pure unitaries) // TODO: should map mlir::Value to mlir::Value. should include getters for targets and controls.
- `hasStaticUnitary() -> bool`
- `tryGetStaticMatrix() -> Optional<Attribute>` (2D tensor with shape (2^n, 2^n) and element type `complex<f64>`; written concretely for fixed n as e.g. `tensor<4x4xcomplex<f64>>`)
- `isInverted() -> bool`
- `getPower() -> Optional<RationalOrFloat>`

Identification & Descriptor Tuple:
`(baseSymbol, orderedParams, posControls, negControls, powerExponent, invertedFlag)` allows canonical equality tests.

Parameter Model:

- Parameters appear in parentheses immediately after mnemonic.
- Mixed static (attributes) and dynamic (operands) preserve original order.
- Enumeration returns flattened ordered list; inspect each for static/dynamic.

Static Matrix Extraction:

- Provided if gate is analytic, all parameters are static, or for matrix-defined user gates.
- For sequences/composites of static subunits, compose matrices.

Inversion & Power Interaction:

- `inv` introduces `invertedFlag` (final canonical position).
- `pow` stores exponent; negative exponent canonicalized to `inv(pow(+exp))` then reordered.

Control Extension:

- `ctrl` / `negctrl` wrappers extend control sets; interface aggregates flattened sets.

## 4. Base Gate Operations

### 4.1 Philosophy

- Named base gates define unitaries with fixed target arity and parameter arity traits.
- Provide static matrix when parameters static or no parameters; symbolic otherwise.
- Avoid embedding modifier semantics directly—wrappers handle extension.

### 4.2 Base Gate Specification Template

For every named base gate op G:

// TODO: mixed dynamic and static parameters should be clearly explained and demonstrated here. example from existing code: `mqtref.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, false, true]) %q0`

- Purpose: Apply the unitary for gate G to its target qubit(s).
- Signature (Reference): `mqtref.G(param_list?) %q[,...] : <param types..., qubit types>` (no results)
- Signature (Value): `%out_targets = mqtopt.G(param_list?) %in_targets : (<param types..., qubit types>) -> <qubit types>`
- Assembly Format: `G(<params?>) <targets>`; params in parentheses; qubits as trailing operands.
- Builder Variants: // TODO: types should be inferred automatically based on `InferTypeOpInterface`
  - `build(builder, loc, resultTypes, paramOperands, qubitOperands)` (value)
  - `build(builder, loc, qubitOperands, paramAttrs)` (reference)
  - Convenience: static param overloads generate attribute parameters.
- Interface Implementation Notes:
  - `getNumTargets()` fixed by trait.
  - Parameters enumerated in declared order; static vs dynamic via attribute vs operand.
  - `hasStaticUnitary()` true iff all parameters static.
  - `mapOutputToInput(i) = i`. // TODO
- Conversion (ref↔value): Reference variant lowers to value variant with SSA replacement; reverse drops result.

### 4.3 Gate List

Overview:

Zero-qubit gates:
Parametrized: `gphase(%theta)`

Single-qubit gates:
No-parameter: `id, x, y, z, h, s, sdg, t, tdg, sx, sxdg`
Parameterized: `rx(%theta), ry(%theta), rz(%theta), p(%lambda), r(theta, %phi), u(%theta, %phi, %lambda), u2(%phi, %lambda)`

Two-qubit gates:
No-parameter: `swap, iswap, dcx, ecr`
Parameterized: `rxx(%theta), ryy(%theta), rzz(%theta), rzx(%theta), xx_minus_yy(%theta, %beta), xx_plus_yy(%theta, %beta)`

Variable qubit gates:
No-parameter: `barrier`

General canonicalization based on traits (not repeated for individual gates):

- Hermitian: `inv G → G`
- Hermitian: `pow(n: int) G => if n % 2 == 0 then id else G`
- Hermitian: `G %q; G %q => cancel`

#### 4.3.1 `gphase` Gate

- Purpose: global phase `exp(i θ)`.
- Traits: NoTarget, OneParameter
- Signatures:
  - Ref: `mqtref.gphase(%theta)`
  - Value: `mqtopt.gphase(%theta)`
- Static Example: `mqtref.gphase(3.14159)`
- Dynamic Example: `mqtref.gphase(%theta)`
- Canonicalization:
  - `gphase(0) → remove`
  - `inv gphase(θ) → gphase(-θ)`
  - Two consecutive `gphase(a); gphase(b)` folded by adding angles.
  - `ctrl(%q0) { gphase(θ) } → p(θ) %q0` (specialization to `p` gate).
  - `negctrl(%q0) { gphase(θ) } → gphase(pi); p(θ) %q0` (specialization for negative control)
  - `pow(n) { gphase(θ) } → gphase(n*θ)`
- Matrix (dynamic): `[exp(i θ)]` (1x1 matrix). Static if θ constant.
- To be figured out: These gates have no users per definition as they have no targets. It is unclear how they should be merged and how they are included in traversals.

### 4.3.2 `id` Gate

- Purpose: Identity gate.
- Traits: OneTarget, NoParameter, Hermitian, Diagonal
- Signatures:
  - Ref: `mqtref.id %q`
  - Value: `%q_out mqtopt.id %q_in`
- Canonicalization:
  - `id → remove`
  - `pow(r) id => id`
  - `ctrl(...) { id } => id`
  - `negctrl(...) { id } => id`
- Matrix (static): `[1, 0; 0, 1]` (2x2 matrix).
- Definition in terms of `u`: `u(0, 0, 0) %q`

#### 4.3.3 `x` Gate

- Purpose: Pauli-X gate
- Traits: OneTarget, NoParameter, Hermitian
- Signatures:
  - Ref: `mqtref.x %q`
  - Value: `%q_out mqtopt.x %q_in`
- Canonicalization:
  - `pow(1/2) x => sx`
  - `pow(-1/2) x => sxdg`
- Matrix (static): `[0, 1; 1, 0]` (2x2 matrix).
- Definition in terms of `u`: `u(π, 0, π) %q`
- To be figured out: `-iX == rx(pi)` (global phase difference between `rx(pi)` and `x`). `pow(r) rx(θ) => rx(r*θ)`. What does this imply for `pow(r) x`?

#### 4.3.4 `y` Gate

- Purpose: Pauli-Y gate
- Traits: OneTarget, NoParameter, Hermitian
- Signatures:
  - Ref: `mqtref.y %q`
  - Value: `%q_out mqtopt.y %q_in`
- Matrix (static): `[0, -i; i, 0]` (2x2 matrix).
- Definition in terms of `u`: `u(π, π/2, π/2) %q`
- To be figured out: `-iY == ry(pi)` (global phase difference between `ry(pi)` and `y`). `pow(r) ry(θ) => ry(r*θ)`. What does this imply for `pow(r) y`?

#### 4.3.5 `z` Gate

- Purpose: Pauli-Z gate
- Traits: OneTarget, NoParameter, Hermitian, Diagonal
- Signatures:
  - Ref: `mqtref.z %q`
  - Value: `%q_out mqtopt.z %q_in`
- Canonicalization:
  - `pow(1/2) z => s`
  - `pow(-1/2) z => sdg`
  - `pow(1/4) z => t`
  - `pow(-1/4) z => tdg`
  - `pow(r) z => p(π * r)` for real r
- Matrix (static): `[1, 0; 0, -1]` (2x2 matrix).
- Definition in terms of `u`: `u(0, 0, π) %q`

#### 4.3.6 `h` Gate

- Purpose: Hadamard gate.
- Traits: OneTarget, NoParameter, Hermitian
- Signatures:
  - Ref: `mqtref.h %q`
  - Value: `%q_out mqtopt.h %q_in`
- Matrix (static): `1/sqrt(2) * [1, 1; 1, -1]` (2x2 matrix).
- Definition in terms of `u`: `u(π/2, 0, π) %q`

#### 4.3.7 `s` Gate

- Purpose: S gate.
- Traits: OneTarget, NoParameter, Diagonal
- Signatures:
  - Ref: `mqtref.s %q`
  - Value: `%q_out mqtopt.s %q_in`
- Canonicalization:
  - `inv s => sdg`
  - `s %q; s %q => z %q`
  - `pow(n: int) s => if n % 4 == 0 then id else if n % 4 == 1 then s else if n % 4 == 2 then z else sdg`
  - `pow(1/2) s => t`
  - `pow(-1/2) s => tdg`
  - `pow(+-2) s => z`
  - `pow(r) s => p(π/2 * r)` for real r
- Matrix (static): `[1, 0; 0, i]` (2x2 matrix).
- Definition in terms of `u`: `u(0, 0, π/2) %q`

#### 4.3.8 `sdg` Gate

- Purpose: Sdg gate.
- Traits: OneTarget, NoParameter, Diagonal
- Signatures:
  - Ref: `mqtref.sdg %q`
  - Value: `%q_out mqtopt.sdg %q_in`
- Canonicalization:
  - `inv sdg => s`
  - `sdg %q; sdg %q => z %q`
  - `pow(n: int) sdg => if n % 4 == 0 then id else if n % 4 == 1 then sdg else if n % 4 == 2 then z else s`
  - `pow(1/2) sdg => tdg`
  - `pow(-1/2) sdg => t`
  - `pow(+-2) sdg => z`
  - `pow(r) sdg => p(-π/2 * r)` for real r
- Matrix (static): `[1, 0; 0, -i]` (2x2 matrix).
- Definition in terms of `u`: `u(0, 0, -π/2) %q`

#### 4.3.9 `t` Gate

- Purpose: T gate.
- Traits: OneTarget, NoParameter, Diagonal
- Signatures:
  - Ref: `mqtref.t %q`
  - Value: `%q_out mqtopt.t %q_in`
- Canonicalization:
  - `inv t => tdg`
  - `t %q; t %q; => s %q`
  - `pow(2) t => s`
  - `pow(-2) t => sdg`
  - `pow(+-4) t => z`
  - `pow(r) t => p(π/4 * r)` for real r
- Matrix (static): `[1, 0; 0, exp(i π/4)]` (2x2 matrix).
- Definition in terms of `u`: `u(0, 0, π/4) %q`

#### 4.3.10 `tdg` Gate

- Purpose: Tdg gate.
- Traits: OneTarget, NoParameter, Diagonal
- Signatures:
  - Ref: `mqtref.tdg %q`
  - Value: `%q_out mqtopt.tdg %q_in`
- Canonicalization:
  - `inv tdg => t`
  - `tdg %q; tdg %q; => sdg %q`
  - `pow(2) tdg => sdg`
  - `pow(-2) tdg => s`
  - `pow(+-4) tdg => z`
  - `pow(r) tdg => p(-π/4 * r)` for real r
- Matrix (static): `[1, 0; 0, exp(-i π/4)]` (2x2 matrix).
- Definition in terms of `u`: `u(0, 0, -π/4) %q`

#### 4.3.11 `sx` Gate

- Purpose: sqrt(x) gate.
- Traits: OneTarget, NoParameter
- Signatures:
  - Ref: `mqtref.sx %q`
  - Value: `%q_out mqtopt.sx %q_in`
- Canonicalization:
  - `inv sx => sxdg`
  - `sx %q; sx %q => x %q`
  - `pow(+-2) sx => x`
- Matrix (static): `1/2 * [1 + i, 1 - i; 1 - i, 1 + i]` (2x2 matrix).
- To be figured out: `e^(-i pi/4) sx == rx(pi/2)` (global phase difference between `rx(pi/2)` and `sx`). `pow(r) rx(θ) => rx(r*θ)`. What does this imply for `pow(r) sx`?

#### 4.3.12 `sxdg` Gate

- Purpose: sqrt(x) gate.
- Traits: OneTarget, NoParameter
- Signatures:
  - Ref: `mqtref.sxdg %q`
  - Value: `%q_out mqtopt.sxdg %q_in`
- Canonicalization:
  - `inv sxdg => sx`
  - `sxdg %q; sxdg %q => x %q`
  - `pow(+-2) sxdg => x`
- Matrix (static): `1/2 * [1 - i, 1 + i; 1 + i, 1 - i]` (2x2 matrix).
- To be figured out: `exp(-i pi/4) sxdg == rx(-pi/2)` (global phase difference between `rx(-pi/2)` and `sxdg`). `pow(r) rx(θ) => rx(r*θ)`. What does this imply for `pow(r) sxdg`?

#### 4.3.13 `rx` Gate

- Purpose: Rotation around the x-axis.
- Traits: OneTarget, OneParameter
- Signatures:
  - Ref: `mqtref.rx(%theta) %q`
  - Value: `%q_out mqtopt.rx(%theta) %q_in`
- Static variant: `mqtref.rx(3.14159) %q`
- Canonicalization:
  - `rx(a) %q; rx(b) %q => rx(a + b) %q`
  - `inv rx(θ) => rx(-θ)`
  - `pow(r) rx(θ) => rx(r * θ)` for real r
- Matrix (dynamic): `exp(-i θ X) = [cos(θ/2), -i sin(θ/2); -i sin(θ/2), cos(θ/2)]` (2x2 matrix). Static if θ constant.
- Definition in terms of `u`: `u(θ, -π/2, π/2) %q`

#### 4.3.14 `ry` Gate

- Purpose: Rotation around the y-axis.
- Traits: OneTarget, OneParameter
- Signatures:
  - Ref: `mqtref.ry(%theta) %q`
  - Value: `%q_out mqtopt.ry(%theta) %q_in`
- Static variant: `mqtref.ry(3.14159) %q`
- Canonicalization:
  - `ry(a) %q; ry(b) %q => ry(a + b) %q`
  - `inv ry(θ) => ry(-θ)`
  - `pow(r) ry(θ) => ry(r * θ)` for real r
- Matrix (dynamic): `exp(-i θ Y) = [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]` (2x2 matrix). Static if θ constant.
- Definition in terms of `u`: `u(θ, 0, 0) %q`

#### 4.3.15 `rz` Gate

- Purpose: Rotation around the z-axis.
- Traits: OneTarget, OneParameter, Diagonal
- Signatures:
  - Ref: `mqtref.rz(%theta) %q`
  - Value: `%q_out mqtopt.rz(%theta) %q_in`
- Static variant: `mqtref.rz(3.14159) %q`
- Canonicalization:
  - `rz(a) %q; rz(b) %q => rz(a + b) %q`
  - `inv rz(θ) => rz(-θ)`
  - `pow(r) rz(θ) => rz(r * θ)` for real r
- Matrix (dynamic): `exp(-i θ Z) = [exp(-i θ/2), 0; 0, exp(i θ/2)]` (2x2 matrix). Static if θ constant.
- To be figured out: `rz(θ) == exp(i*θ/2) * p(θ)` (global phase difference between `rz(θ)` and `p(θ)`).

#### 4.3.16 `p` Gate

- Purpose: Phase gate.
- Traits: OneTarget, OneParameter, Diagonal
- Signatures:
  - Ref: `mqtref.p(%theta) %q`
  - Value: `%q_out mqtopt.p(%theta) %q_in`
- Static variant: `mqtref.p(3.14159) %q`
- Canonicalization:
  - `p(a) %q; p(b) %q => p(a + b) %q`
  - `inv p(θ) => p(-θ)`
  - `pow(r) p(θ) => p(r * θ)` for real r
- Matrix (dynamic): `[1, 0; 0, exp(i θ)]` (2x2 matrix). Static if θ constant.
- Definition in terms of `u`: `u(0, 0, θ) %q`

#### 4.3.17 `r` Gate

- Purpose: General rotation around an axis in the XY-plane.
- Traits: OneTarget, TwoParameter
- Signatures:
  - Ref: `mqtref.r(%theta, %phi) %q`
  - Value: `%q_out mqtopt.r(%theta, %phi) %q_in`
- Static variant: `mqtref.r(3.14159, 1.5708) %q`
- Mixed variant: `mqtref.r(%theta, 1.5708) %q`
- Canonicalization:
  - `inv r(θ, φ) => r(-θ, φ)`
  - `pow(real) r(θ, φ) => r(real * θ, φ)` for real `real`
  - `r(θ, 0) => rx(θ)`
  - `r(θ, π/2) => ry(θ)`
- Matrix (dynamic): `exp(-i θ (cos(φ) X + sin(φ) Y)) = [cos(θ/2), -i exp(-i φ) sin(θ/2); -i exp(i φ) sin(θ/2), cos(θ/2)]` (2x2 matrix). Static if θ and φ constant.
- Definition in terms of `u`: `u(θ, -π/2 + φ, π/2 - φ) %q`

#### 4.3.18 `u` Gate

- Purpose: Universal single-qubit gate.
- Traits: OneTarget, ThreeParameter
- Signatures:
  - Ref: `mqtref.u(%theta, %phi, %lambda) %q`
  - Value: `%q_out mqtopt.u(%theta, %phi, %lambda) %q_in`
- Static variant: `mqtref.u(3.14159, 1.5708, 0.785398) %q`
- Mixed variant: `mqtref.u(%theta, 1.5708, 0.785398) %q`
- Canonicalization:
  - `inv u(θ, φ, λ) => u(-θ, -φ, -λ)`
  - `rx(θ) == u(θ, -π/2, π/2)`
  - `ry(θ) == u(θ, 0, 0)`
  - `p(λ) == u(0, 0, λ)`
- Matrix (dynamic): `p(φ) ry(θ) p(λ) = exp(i (φ + λ)/2) * rz(φ) ry(θ) rz(λ) = [cos(θ/2), -exp(i λ) sin(θ/2); exp(i φ) sin(θ/2), exp(i (φ + λ)) cos(θ/2)]` (2x2 matrix). Static if θ, φ, λ constant.

#### 4.3.19 `u2` Gate

- Purpose: Simplified universal single-qubit gate.
- Traits: OneTarget, TwoParameter
- Signatures
  - Ref: `mqtref.u2(%phi, %lambda) %q`
  - Value: `%q_out mqtopt.u2(%phi, %lambda) %q_in`
- Static variant: `mqtref.u2(1.5708, 0.785398) %q`
- Mixed variant: `mqtref.u2(%phi, 0.785398) %q`
- Canonicalization:
  - `inv u2(φ, λ) => u2(-λ - π, -φ + π)`
  - `u2(0, π) => h`
  - `u2(0, 0) => ry(π/2)`
  - `u2(-π/2, π/2) => rx(π/2)`
- Matrix (dynamic): `1/sqrt(2) * [1, -exp(i λ); exp(i φ), exp(i (φ + λ))]` (2x2 matrix). Static if φ, λ constant.
- Definition in terms of `u`: `u2(φ, λ) == u(π/2, φ, λ)`

#### 4.3.20 `swap` Gate

- Purpose: Swap two qubits.
- Traits: TwoTarget, NoParameter, Hermitian
- Signatures:
  - Ref: `mqtref.swap %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.swap %q0_in, %q1_in`
- Matrix (static): `[1, 0, 0, 0; 0, 0, 1, 0; 0, 1, 0, 0; 0, 0, 0, 1]` (4x4 matrix).

#### 4.3.21 `iswap` Gate

- Purpose: Swap two qubits.
- Traits: TwoTarget, NoParameter
- Signatures:
  - Ref: `mqtref.iswap %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.iswap %q0_in, %q1_in`
- Canonicalization:
  - `pow(r) iswap => xx_plus_yy(-π * r)`
- Matrix (static): `[1, 0, 0, 0; 0, 0, 1j, 0; 0, 1j, 0, 0; 0, 0, 0, 1]` (4x4 matrix).

#### 4.3.22 `dcx` Gate

- Purpose: Double CX gate.
- Traits: TwoTarget, NoParameter
- Signatures:
  - Ref: `mqtref.dcx %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.dcx %q0_in, %q1_in`
- Canonicalization:
  - `inv dcx %q0, q1 => dcx %q1, %q0`
- Matrix (static): `[1, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 1, 0, 0]` (4x4 matrix).`

#### 4.3.23 `ecr` Gate

- Purpose: Echoed cross-resonance gate.
- Traits: TwoTarget, NoParameter, Hermitian
- Signatures:
  - Ref: `mqtref.ecr %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.ecr %q0_in, %q1_in`
- Matrix (static): `1/sqrt(2) * [0, 0, 1, 1j; 0, 0, 1j, 1; 1, -1j, 0, 0; -1j, 1, 0, 0]` (4x4 matrix).

#### 4.3.24 `rxx` Gate

- Purpose: General two-qubit rotation around XX.
- Traits: TwoTarget, OneParameter
- Signatures:
  - Ref: `mqtref.rxx(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.rxx(%theta) %q0_in, %q1_in`
- Static variant: `mqtref.rxx(3.14159) %q0, %q1`
- Canonicalization:
  - `inv rxx(%theta) => rxx(-%theta)`
  - `pow(r) rxx(%theta) => rxx(r * %theta)` for real r
  - `rxx(0) => remove`
  - `rxx(a) %q0, %q1; rxx(b) %q0, %q1 => rxx(a + b) %q0, %q1`
- Matrix (dynamic): `cos(θ/2) * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] - 1j * sin(θ/2) * [0, 0, 0, 1; 0, 0, 1, 0; 0, 1, 0, 0; 1, 0, 0, 0]` (4x4 matrix). Static if θ constant.

#### 4.3.25 `ryy` Gate

- Purpose: General two-qubit gate around YY.
- Traits: TwoTarget, OneParameter
- Signatures:
  - Ref: `mqtref.ryy(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.ryy(%theta) %q0_in, %q1_in`
- Static variant: `mqtref.ryy(3.14159) %q0, %q1`
- Canonicalization:
  - `inv ryy(%theta) => ryy(-%theta)`
  - `pow(r) ryy(%theta) => ryy(r * %theta)` for real r
  - `ryy(0) => remove`
  - `ryy(a) %q0, %q1; ryy(b) %q0, %q1 => ryy(a + b) %q0, %q1`
- Matrix (dynamic): `cos(θ/2) * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] + 1j * sin(θ/2) * [0, 0, 0, 1; 0, 0, -1, 0; 0, -1, 0, 0; 1, 0, 0, 0]` (4x4 matrix). Static if θ constant.

#### 4.3.26 `rzx` Gate

- Purpose: General two-qubit gate around ZX.
- Traits: TwoTarget, OneParameter
- Signatures:
  - Ref: `mqtref.rzx(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.rzx(%theta) %q0_in, %q1_in`
- Static variant: `mqtref.rzx(3.14159) %q0, %q1`
- Canonicalization:
  - `inv rzx(%theta) => rzx(-%theta)`
  - `pow(r) rzx(%theta) => rzx(r * %theta)` for real r
  - `rzx(0) => remove`
  - `rzx(a) %q0, %q1; rzx(b) %q0, %q1 => rzx(a + b) %q0, %q1`
- Matrix (dynamic): `cos(θ/2) * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] + 1j * sin(θ/2) * [0, -1, 0, 0; -1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0]` (4x4 matrix). Static if θ constant.

#### 4.3.27 `rzz` Gate

- Purpose: General two-qubit gate around ZZ.
- Traits: TwoTarget, OneParameter, Diagonal
- Signatures:
  - Ref: `mqtref.rzz(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.rzz(%theta) %q0_in, %q1_in`
- Static variant: `mqtref.rzz(3.14159) %q0, %q1`
- Canonicalization:
  - `inv rzz(%theta) => rzz(-%theta)`
  - `pow(r) rzz(%theta) => rzz(r * %theta)` for real r
  - `rzz(0) => remove`
  - `rzz(a) %q0, %q1; rzz(b) %q0, %q1 => rzz(a + b) %q0, %q1`
- Matrix (dynamic): `diag[exp(-i θ/2), exp(i θ/2), exp(i θ/2), exp(-i θ/2)]` (4x4 matrix). Static if θ constant.

#### 4.3.28 `xx_plus_yy` Gate

- Purpose: General two-qubit gate around XX+YY.
- Traits: TwoTarget, TwoParameter
- Signatures:
  - Ref: `mqtref.xx_plus_yy(%theta, %beta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.xx_plus_yy(%theta, %beta) %q0_in, %q1_in`
- Static variant: `mqtref.xx_plus_yy(3.14159, 1.5708) %q0, %q1`
- Mixed variant: `mqtref.xx_plus_yy(%theta, 1.5708) %q0, %q1`
- Canonicalization:
  - `inv xx_plus_yy(θ, β) => xx_plus_yy(-θ, β)`
  - `pow(r) xx_plus_yy(θ, β) => xx_plus_yy(r * θ, β)` for real r
  - `xx_plus_yy(θ1, β) %q0, %q1; xx_plus_yy(θ2, β) %q0, %q1 => xx_plus_yy(θ1 + θ2, β) %q0, %q1`
- Matrix (dynamic): `[1, 0, 0, 0; 0, cos(θ/2), sin(θ/2) * exp(-i β), 0; 0, -sin(θ/2) * exp(i β), cos(θ/2), 0; 0, 0, 0, 1]` (4x4 matrix). Static if θ and β constant.

#### 4.3.29 `xx_minus_yy` Gate

- Purpose: General two-qubit gate around XX-YY.
- Traits: TwoTarget, TwoParameter
- Signatures:
  - Ref: `mqtref.xx_minus_yy(%theta, %beta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.xx_minus_yy(%theta, %beta) %q0_in, %q1_in`
- Static variant: `mqtref.xx_minus_yy(3.14159, 1.5708) %q0, %q1`
- Mixed variant: `mqtref.xx_minus_yy(%theta, 1.5708) %q0, %q1`
- Canonicalization:
  - `inv xx_minus_yy(θ, β) => xx_minus_yy(-θ, β)`
  - `pow(r) xx_minus_yy(θ, β) => xx_minus_yy(r * θ, β)` for real r
  - `xx_minus_yy(θ1, β) %q0, %q1; xx_minus_yy(θ2, β) %q0, %q1 => xx_minus_yy(θ1 + θ2, β) %q0, %q1`
- Matrix (dynamic): `[cos(θ/2), 0, 0, -sin(θ/2) * exp(i β); 0, 1, 0, 0; 0, 0, 1, 0; sin(θ/2) * exp(-i β), 0, 0, cos(θ/2)]` (4x4 matrix). Static if θ and β constant.

### 5. Modifier Operations

### 5.1 Overview

Modifiers wrap unitaries, extending functionality or altering semantics.
They contain exactly one region with a single block whose only operation implements the `UnitaryOpInterface`.
In the reference semantics dialect, modifiers are statements without results.
In the value semantics dialect, modifiers thread their values through region arguments and yield results.
Converting from the value semantics to the reference semantics is straightforward.
The reverse direction requires a bit of care as the SSA values of the contained unitary need to be added to the region arguments as well as the results need to be yielded.
Modifiers may be arbitrarily nested, with canonicalization rules to flatten and reorder them.

There are three types of modifiers:

- Control modifiers: `ctrl` and `negctrl`. These add additional (control) qubits to an operation. They extend the qubit list of the unitary operation in question.
- Inverse modifier: `inv`. This takes the adjoint of the unitary operation. Specializations for many of the basis gates exist and are defined as canonicalization rules.
- Power modifier: `pow`. This takes the power of the unitary operation. Canonicalization rules are provided to simplify common cases.

The canonical ordering for these modifiers is (from outside to inside): `negtrcl` -> `ctrl` -> `pow` -> `inv`.

All modifiers share a common verifier: they must have a single block with a single operation implementing the `UnitaryOpInterface`.

### 5.2 Control Modifiers

- Purpose: Add additional (control) qubits to an operation. Control qubits can either be positive (1-state) or negative (0-state) controls. The modifier itself holds a variadic list of qubits.
- Signatures (just shown for `ctrl` for simplicity):
  - Ref: `mqtref.ctrl(%ctrls) { mqtref.unitaryOp %targets }`
  - Value:
    ```
    %ctrl_outs, %unitary_outs = mqtopt.ctrl(%ctrl_ins, %unitary_ins) {
      %u_outs = mqtopt.unitaryOp %unitary_ins
      mqtopt.yield %u_outs
    }
    ```
- Builders: Provide list of qubits + body builder.
- Interface:
  - Targets: targets of child unitary
  - Controls: controls of modifier plus controls of child unitary
  - Parameters: aggregated from child unitary (none directly)
  - `hasStaticUnitary()` if child unitary static
- Canonicalization:
  - Flatten nested control modifiers by merging control lists.
  - Remove empty control modifiers.
  - Controls applied to global phase gate +> pick one (arbitrary control) and replace the global phase gate with a (controlled) phase gate.
  - Canonical modifier ordering:
    - `ctrl negctrl U => negctrl ctrl U`
- Verifiers:
  - Ensure control and target qubits are distinct.
- Unitary computation: Computed by expanding the unitary of the child operation to the larger space defined by the additional control qubits.

### 5.3 Inverse Modifier

- Purpose: Take the adjoint of the unitary operation.
- Signatures:
  - Ref: `mqtref.inv { mqtref.unitaryOp %targets }`
  - Value:
    ```
    %unitary_outs = mqtopt.inv(%unitary_ins) {
      %u_outs = mqtopt.unitaryOp %unitary_ins
      mqtopt.yield %u_outs
    }
    ```
- Builders: Provide body builder.
- Interface:
  - Targets: targets of child unitary
  - Controls: controls of child unitary
  - Parameters: aggregated from child unitary (none directly)
  - `hasStaticUnitary()` if child unitary static
- Canonicalization:
  - Pairs of nested inverses cancel, i.e. `inv inv U => U`.
  - Specializations for many basis gates exist and are defined as canonicalization rules.
  - Canonical modifier ordering:
    - `inv ctrl U => ctrl inv U`
    - `inv negctrl U => negctrl inv U`
- Verifiers: None additional.
- Unitary computation: Computed by inverting the unitary of the child operation. Given how the underlying operation is unitary, the inverse is given by the conjugate transpose.

### 5.4 Power Modifier

- Purpose: Take the power of the unitary operation.
- Signatures:
  - Ref: `mqtref.pow(%exponent) { mqtref.unitaryOp %targets }`
  - Value:
    ```
    %unitary_outs = mqtopt.pow(%exponent, %unitary_ins) {
      %u_outs = mqtopt.unitaryOp %unitary_ins
      mqtopt.yield %u_outs
    }
    ```
- Static variant: `mqtref.pow(3) { mqtref.unitaryOp %targets }`
- Builders: Provide exponent value (or attribute) + body builder.
- Interface:
  - Targets: targets of child unitary
  - Controls: controls of child unitary
  - Parameters: aggregated from child unitary + exponent (either counted as static or dynamic parameter)
  - `hasStaticUnitary()` if child unitary static and exponent static
- Canonicalization:
  - Flatten nested power modifiers by multiplying exponents.
  - Remove power modifier with exponent 1.
  - `pow(0) U => remove` completely removed the modifier and the operation
  - Specializations for many basis gates exist and are defined as canonicalization rules.
  - Constant folding and propagation of exponents, e.g., replacing constant values by attributes.
  - Negative exponents are pushed into the child unitary by inverting it, e.g., `pow(-r) U => pow(r) inv(U)`.
  - Canonical modifier ordering:
    - `pow ctrl U => ctrl pow U`
    - `pow negctrl U => negctrl pow U`
- Verifiers: None additional.
- Unitary computation: Computed by raising the unitary of the child operation to the given power. For positive integer exponents, this is simply a matrix multiplication. For real-valued exponents, this can be computed by exponentiation.

## 6. Sequence Operation (`seq`)

- Purpose: Ordered, unnamed composition of unitary operations over region block arguments.
- Signatures:
  - Ref: `mqtref.seq { <unitary-ops> }`
  - Value:
    ```
    %results = mqtopt.seq(%args) -> (%result_types) {
      <ops>
      mqtopt.yield %newArgs
    }
    ```
- Builders: Provide body builder.
- Interface:
  - Targets = Aggregated targets of child unitary ops (none directly)
  - Controls = None
  - Parameters = Aggregated parameters of child unitary ops (none directly)
  - `hasStaticUnitary()` if all child ops static`
- Canonicalization:
  - Remove empty sequence.
  - Replace sequence with a single operation by inlining that operation.
  - Provide inlining capabilities for flattening nested sequences.
- Verifiers: All block operations must implement the `UnitaryOpInterface`.
- Unitary computation: Computed by computing the product of the unitaries of the child operations, i.e., `U_n U_{n-1} ... U_1 U_0`.
- Conversion:
  - Value semantics to reference semantics: Remove block arguments and results, replace uses of arguments with direct uses of the corresponding values.
  - Reference semantics to value semantics: Add block arguments and results, replace direct use of values with uses of the corresponding arguments.

## 7. User Defined Gates & Matrix / Composite Definitions

In addition to the unnamed sequence operation, the dialect also provides a mechanism for defining custom (unitary) gates that produce a symbol that can be referenced in an `apply` operation.
Conceptionally, this should be very close to an actual MLIR function definition and call.

These gates can be defined in two ways:

- Matrix-based definition: Define a gate using a matrix representation.
- Sequence-based definition: Define a gate using a sequence of (unitary) operations.

The matrix-based definition is very efficient for small qubit numbers, but does not scale well for larger numbers of qubits.
The sequence-based definition is more general, but requires more processing to compute the underlying functionality.
Definitions might even provide both a matrix and a sequence representation, which should be consistent.

Matrix-based definitions may be fully static or might be based on symbolic expressions, e.g., rotation gates.
Fully static matrices may be specified as dense array attributes.
Any dynamic definitions must be (somehow) specified as symbolic expressions.

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
