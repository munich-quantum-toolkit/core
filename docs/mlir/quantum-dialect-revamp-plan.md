# Quantum Dialect Revamp RFC

## Executive Summary

This RFC proposes a comprehensive redesign of the MQT quantum MLIR dialect to provide a unified, extensible framework for quantum circuit representation and optimization.
The revamp introduces a dual-dialect approach (`mqtref` for reference semantics, `mqtopt` for value semantics), a unified unitary interface, composable modifiers (control, inversion, power), user-defined gates, and a robust canonicalization framework.

**Key Benefits:**

- **Unified Interface:** All unitary operations expose consistent APIs for introspection and composition
- **Enhanced Expressiveness:** Support for arbitrary gate modifications, custom gate definitions, and symbolic parameters
- **Optimization-Ready:** Built-in canonicalization rules and transformation hooks
- **Dual Semantics:** Choose between reference semantics (hardware-like) or value semantics (SSA-based optimization)

## 1. Overview and Goals

This RFC proposes a comprehensive revamp of the quantum MLIR dialect(s) to unify unitary representations, improve expressiveness, and enable robust transformations.

**Primary Goals:**

- **Unified Unitary Interface:** Provide a coherent interface for all operations that apply or produce a unitary (base gates, user-defined gates, modifier-wrapped constructs, sequences)
- **Dual Semantics Support:** Support both reference semantics (in-place: `mqtref`) and value semantics (SSA threading: `mqtopt`)
- **Rich Modifier System:** Add inversion, powering, and positive/negative multi-controls as composable modifiers
- **Custom Gate Support:** Enable user-defined gates via matrix and composite (sequence-based) definitions
- **Consistent Parameterization:** Unify parameter handling (static + dynamic) with consistent ordering and interface queries
- **Canonicalization Framework:** Embed canonicalization rules directly at each operation definition
- **Normalized Modifier Nesting:** Establish a canonical modifier nesting order: `negctrl → ctrl → pow → inv`
- **Matrix Extraction:** Enable static matrix extraction where possible and symbolic composition otherwise

## 2. Current State and Limitations

The current implementation has several significant limitations that this revamp addresses:

**Existing Issues:**

- **Limited Modifiers:** Only a control modifier exists, directly embedded in unitary operations; missing power and inversion modifiers
- **Missing Matrix Support:** No way to obtain matrix representations for gates
- **No Custom Gates:** No support for user-defined gates (neither matrix-based nor composite)
- **Absent Canonicalization:** No systematic canonicalization strategy for modifier order, parameter folding, or gate specialization
- **Testing Challenges:** Mostly FileCheck-based testing that is cumbersome and error-prone to write
- **Limited Builders:** No convenient programmatic builders for constructing quantum programs

These limitations hinder both expressiveness and optimization capabilities, motivating the comprehensive redesign proposed in this RFC.

## 3. Dialect Structure and Categories

The revamp introduces two parallel dialects with identical operation sets but different operational semantics:

### 3.1 Dialect Overview

**`mqtref` (Reference Semantics):**

- Operations mutate qubits in-place (similar to hardware model)
- No SSA results for qubit operations
- More natural for hardware mapping and direct circuit representation
- Example: `mqtref.h %q` applies Hadamard to qubit `%q` in-place

**`mqtopt` (Value Semantics):**

- Operations consume and produce new SSA values (functional style)
- Enables powerful SSA-based optimizations and transformations
- More natural for compiler optimization passes
- Example: `%q_out = mqtopt.h %q_in` consumes `%q_in` and produces `%q_out`

Both dialects share the same operation names and semantics, differing only in their type system and SSA threading model. Conversion passes enable moving between the two dialects as needed.

### 3.2 Operation Categories

All operations fall into three primary categories:

1. **Resource Operations:** Manage qubit lifetime and allocation
2. **Measurement and Reset:** Non-unitary operations that collapse or reinitialize quantum states
3. **Unitary Operations:** All operations implementing the `UnitaryOpInterface` (base gates, modifiers, sequences, custom gates)

### 3.3 Resource Operations

**Purpose:** Manage qubit lifetime and references.

**Reference Semantics (`mqtref`):**

```mlir
%q = mqtref.alloc : !mqtref.qubit
mqtref.dealloc %q : !mqtref.qubit
%q0 = mqtref.qubit 0 : !mqtref.qubit  // Static qubit reference
```

**Value Semantics (`mqtopt`):**

```mlir
%q = mqtopt.alloc : !mqtopt.qubit
mqtopt.dealloc %q : !mqtopt.qubit
%q0 = mqtopt.qubit 0 : !mqtopt.qubit  // Static qubit reference
```

**Canonicalization Patterns:**

- Dead allocation elimination: Remove unused `alloc` operations (DCE)
- **TODO:** Define register allocation operations for multi-qubit arrays
- **TODO:** Specify interaction with classical bit allocation

### 3.4 Measurement and Reset

Non-unitary operations that do not implement the `UnitaryOpInterface`.

**Reference Semantics:**

```mlir
%c = mqtref.measure %q : !mqtref.qubit -> i1
mqtref.reset %q : !mqtref.qubit
```

**Value Semantics:**

```mlir
%q_out, %c = mqtopt.measure %q_in : !mqtopt.qubit -> (!mqtopt.qubit, i1)
%q_out = mqtopt.reset %q_in : !mqtopt.qubit -> !mqtopt.qubit
```

**Canonicalization Patterns:**

- `reset` immediately after `alloc` → remove `reset` (already in ground state)
- Consecutive `reset` on same qubit → single instance
- **TODO:** Specify multi-qubit measurement operations
- **TODO:** Define measurement basis specification (currently assumes computational basis)

### 3.5 Unified Unitary Interface Design

All unitary-applying operations implement a common `UnitaryOpInterface` that provides uniform introspection and composition capabilities. This applies to base gates, modifiers, sequences, and user-defined operations.

**Interface Methods:**

```c++
// Qubit accessors
size_t getNumTargets();
size_t getNumPosControls();
size_t getNumNegControls();
Value getTarget(size_t i);
Value getPosControl(size_t i);
Value getNegControl(size_t i);

// Value semantics threading
Value getInput(size_t i);           // Combined controls + targets
Value getOutput(size_t i);          // Combined controls + targets
Value getOutputForInput(Value in);  // Identity in reference semantics
Value getInputForOutput(Value out); // Identity in reference semantics

// Parameter handling
size_t getNumParams();
ParameterDescriptor getParameter(size_t i);

// Parameter descriptor
struct ParameterDescriptor {
  bool isStatic();                  // True if attribute, false if operand
  Optional<double> getConstantValue();  // If static
  Value getValueOperand();          // If dynamic
};

// Matrix extraction
bool hasStaticUnitary();
Optional<DenseElementsAttr> tryGetStaticMatrix();  // tensor<2^n x 2^n x complex<f64>>

// Modifier state
bool isInverted();
Optional<double> getPower();  // Returns power exponent if applicable

// Identification
std::string getBaseSymbol();
CanonicalDescriptor getCanonicalDescriptor();  // For equivalence testing
```

**Canonical Descriptor Tuple:**

Each unitary can be uniquely identified by the tuple:

```
(baseSymbol, orderedParams, posControls, negControls, powerExponent, invertedFlag)
```

This enables canonical equality tests and efficient deduplication.

**Parameter Model:**

- Parameters appear in parentheses immediately after the operation mnemonic
- Support for mixed static (attributes) and dynamic (SSA values) parameters in original order
- Enumeration returns a flattened ordered list where each parameter can be inspected for static/dynamic nature
- Example: `mqtref.u(%theta, 1.5708, %lambda) %q` has three parameters: dynamic, static, dynamic

**Static Matrix Extraction:**

- Provided when the gate is analytic and all parameters are static
- For matrix-defined user gates, returns the defined matrix
- For sequences/composites of static subunits, composes matrices via multiplication
- Returns `std::nullopt_t` for symbolic or dynamic parameterizations

**Modifier Interaction:**

- `inv` modifier introduces `invertedFlag` in the canonical descriptor
- `pow` modifier stores exponent; negative exponents are canonicalized to `inv(pow(+exp))` then reordered
- Control modifiers (`ctrl`, `negctrl`) extend control sets; interface aggregates flattened sets across nested modifiers

## 4. Base Gate Operations

### 4.1 Philosophy and Design Principles

**Core Principles:**

- **Named Basis Gates:** Each base gate defines a unitary with fixed target arity and parameter arity (expressed via traits)
- **Static Matrix When Possible:** Provide static matrix representations when parameters are static or absent
- **Modifier-Free Core:** Avoid embedding modifier semantics directly—use wrapper operations instead
- **Consistent Signatures:** Maintain uniform syntax across reference and value semantics

**Benefits:**

- Simplifies gate definitions and verification
- Enables powerful pattern matching and rewriting
- Separates concerns between gate semantics and modifications

### 4.2 Base Gate Specification Template

For every named base gate operation `G`:

**Specification Elements:**

- **Purpose:** Brief description of the unitary operation
- **Traits:** Target arity (e.g., `OneTarget`, `TwoTarget`), parameter arity (e.g., `OneParameter`), special properties (e.g., `Hermitian`, `Diagonal`)
- **Signatures:**
  - Reference: `mqtref.G(param_list?) %targets : (param_types..., qubit_types...)`
  - Value: `%out_targets = mqtopt.G(param_list?) %in_targets : (param_types..., qubit_types...) -> (qubit_types...)`
- **Assembly Format:** `G(params?) targets` where params are in parentheses, qubits as trailing operands
- **Interface Implementation:**
  - `getNumTargets()` fixed by target arity trait
  - Parameters enumerated in declared order (static via attribute, dynamic via operand)
  - `hasStaticUnitary()` returns true iff all parameters are static
- **Canonicalization:** List of simplification rules and rewrites
- **Matrix:** Mathematical matrix representation (static or symbolic)
- **Equivalences:** Relationships to other gates (e.g., decomposition in terms of `u` gate)

**General Canonicalization Patterns Based on Traits:**

The following canonicalization patterns apply automatically to all gates with the specified traits (not repeated for individual gates):

- **Hermitian gates:**
  - `inv(G) → G` (self-adjoint)
  - `G %q; G %q → remove` (cancellation)
  - `pow(n: even_integer) G → id`
  - `pow(n: odd_integer) G → G`
- **Diagonal gates:**
  - Commute with other diagonal gates on same qubits
  - Can be merged when adjacent
- **Zero-parameter gates with Hermitian trait:**
  - Consecutive applications cancel

### 4.3 Gate Catalog

**Gate Organization:**

- **Zero-qubit gates:** Global phase
- **Single-qubit gates:** Pauli gates, rotations, phase gates, universal gates
- **Two-qubit gates:** Entangling gates, Ising-type interactions
- **Variable-qubit gates:** Barrier and utility operations

#### 4.3.1 `gphase` Gate (Global Phase)

- **Purpose:** Apply global phase `exp(iθ)` to the quantum state
- **Traits:** `NoTarget`, `OneParameter`
- **Signatures:**
  - Ref: `mqtref.gphase(%theta)`
  - Value: `mqtopt.gphase(%theta)`
- **Examples:**
  - Static: `mqtref.gphase(3.14159)`
  - Dynamic: `mqtref.gphase(%theta)`
- **Canonicalization:**
  - `gphase(0) → remove`
  - `inv(gphase(θ)) → gphase(-θ)`
  - `gphase(a); gphase(b) → gphase(a + b)` (consecutive phases merge)
  - `ctrl(%q) { gphase(θ) } → p(θ) %q` (controlled global phase becomes phase gate)
  - `negctrl(%q) { gphase(θ) } → gphase(π); p(θ) %q` (negative control specialization)
  - `pow(n) { gphase(θ) } → gphase(n*θ)`
- **Matrix:** `[exp(iθ)]` (1×1 scalar, static if θ constant)
- **Open Issues:**
  - **TODO:** Global phase gates have no target qubits, making traversal and merging semantics unclear
  - **TODO:** Define how global phases interact with circuit-level operations
  - **TODO:** Specify whether global phases should be preserved or eliminated in certain contexts

#### 4.3.2 `id` Gate (Identity)

- **Purpose:** Identity operation (does nothing)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.id %q`
  - Value: `%q_out = mqtopt.id %q_in`
- **Canonicalization:**
  - `id → remove` (no effect)
  - `pow(r) id → id` (any power is still id)
  - `ctrl(...) { id } → id` (control with id is just id)
  - `negctrl(...) { id } → id`
- **Matrix:** `[1, 0; 0, 1]` (2x2 identity matrix)
- **Definition in terms of `u`:** `u(0, 0, 0) %q`

#### 4.3.3 `x` Gate (Pauli-X)

- **Purpose:** Pauli-X gate (bit flip)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Ref: `mqtref.x %q`
  - Value: `%q_out = mqtopt.x %q_in`
- **Canonicalization:**
  - `pow(1/2) x → sx` (square root of x is sx)
  - `pow(-1/2) x → sxdg` (inverse square root is sxdg)
- **Matrix:** `[0, 1; 1, 0]` (2x2 matrix)
- **Definition in terms of `u`:** `u(π, 0, π) %q`
- **Open Issues:**
  - **TODO:** Resolve global phase relationship: `-iX == rx(π)`. What are the implications for `pow(r) x`?
  - **TODO:** Define whether `pow(1/3) x` should be supported or rejected

#### 4.3.4 `y` Gate (Pauli-Y)

- **Purpose:** Pauli-Y gate (bit and phase flip)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Ref: `mqtref.y %q`
  - Value: `%q_out = mqtopt.y %q_in`
- **Matrix:** `[0, -i; i, 0]` (2x2 matrix)
- **Definition in terms of `u`:** `u(π, π/2, π/2) %q`
- **Open Issues:**
  - **TODO:** Resolve global phase relationship: `-iY == ry(π)`. What are the implications for `pow(r) y`?

#### 4.3.5 `z` Gate (Pauli-Z)

- **Purpose:** Pauli-Z gate (phase flip)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.z %q`
  - Value: `%q_out = mqtopt.z %q_in`
- **Canonicalization:**
  - `pow(1/2) z → s`
  - `pow(-1/2) z → sdg`
  - `pow(1/4) z → t`
  - `pow(-1/4) z → tdg`
  - `pow(r) z → p(π * r)` for real r
- **Matrix:** `[1, 0; 0, -1]` (2x2 matrix)
- **Definition in terms of `u`:** `u(0, 0, π) %q`

#### 4.3.6 `h` Gate (Hadamard)

- **Purpose:** Hadamard gate (creates superposition)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Ref: `mqtref.h %q`
  - Value: `%q_out = mqtopt.h %q_in`
- **Matrix:** `1/sqrt(2) * [1, 1; 1, -1]` (2x2 matrix)
- **Definition in terms of `u`:** `u(π/2, 0, π) %q`

#### 4.3.7 `s` Gate (S/Phase-90)

- **Purpose:** S gate (applies a phase of π/2)
- **Traits:** `OneTarget`, `NoParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.s %q`
  - Value: `%q_out = mqtopt.s %q_in`
- **Canonicalization:**
  - `inv s → sdg`
  - `s %q; s %q → z %q`
  - `pow(n: int) s → if n % 4 == 0 then id else if n % 4 == 1 then s else if n % 4 == 2 then z else sdg`
  - `pow(1/2) s → t`
  - `pow(-1/2) s → tdg`
  - `pow(+-2) s → z`
  - `pow(r) s → p(π/2 * r)` for real r
- **Matrix:** `[1, 0; 0, i]` (2x2 matrix)
- **Definition in terms of `u`:** `u(0, 0, π/2) %q`

#### 4.3.8 `sdg` Gate (S-Dagger)

- **Purpose:** Sdg gate (applies a phase of -π/2)
- **Traits:** `OneTarget`, `NoParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.sdg %q`
  - Value: `%q_out = mqtopt.sdg %q_in`
- **Canonicalization:**
  - `inv sdg → s`
  - `sdg %q; sdg %q → z %q`
  - `pow(n: int) sdg → if n % 4 == 0 then id else if n % 4 == 1 then sdg else if n % 4 == 2 then z else s`
  - `pow(1/2) sdg → tdg`
  - `pow(-1/2) sdg → t`
  - `pow(+-2) sdg → z`
  - `pow(r) sdg → p(-π/2 * r)` for real r
- **Matrix:** `[1, 0; 0, -i]` (2x2 matrix)
- **Definition in terms of `u`:** `u(0, 0, -π/2) %q`

#### 4.3.9 `t` Gate (T/π-8)

- **Purpose:** T gate (applies a phase of π/4)
- **Traits:** `OneTarget`, `NoParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.t %q`
  - Value: `%q_out = mqtopt.t %q_in`
- **Canonicalization:**
  - `inv t → tdg`
  - `t %q; t %q; → s %q`
  - `pow(2) t → s`
  - `pow(-2) t → sdg`
  - `pow(+-4) t → z`
  - `pow(r) t → p(π/4 * r)` for real r
- **Matrix:** `[1, 0; 0, exp(i π/4)]` (2x2 matrix)
- **Definition in terms of `u`:** `u(0, 0, π/4) %q`

#### 4.3.10 `tdg` Gate (T-Dagger)

- **Purpose:** Tdg gate (applies a phase of -π/4)
- **Traits:** `OneTarget`, `NoParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.tdg %q`
  - Value: `%q_out = mqtopt.tdg %q_in`
- **Canonicalization:**
  - `inv tdg → t`
  - `tdg %q; tdg %q; → sdg %q`
  - `pow(2) tdg → sdg`
  - `pow(-2) tdg → s`
  - `pow(+-4) tdg → z`
  - `pow(r) tdg → p(-π/4 * r)` for real r
- **Matrix:** `[1, 0; 0, exp(-i π/4)]` (2x2 matrix)
- **Definition in terms of `u`:** `u(0, 0, -π/4) %q`

#### 4.3.11 `sx` Gate (√X)

- **Purpose:** Square root of X gate
- **Traits:** `OneTarget`, `NoParameter`
- **Signatures:**
  - Ref: `mqtref.sx %q`
  - Value: `%q_out = mqtopt.sx %q_in`
- **Canonicalization:**
  - `inv sx → sxdg`
  - `sx %q; sx %q → x %q`
  - `pow(+-2) sx → x`
- **Matrix:** `1/2 * [1 + i, 1 - i; 1 - i, 1 + i]` (2x2 matrix)
- **Open Issues:**
  - **TODO:** Resolve global phase relationship: `exp(-iπ/4) sx == rx(π/2)`. Define power semantics.

#### 4.3.12 `sxdg` Gate (√X-Dagger)

- **Purpose:** Inverse of the square root of X gate
- **Traits:** `OneTarget`, `NoParameter`
- **Signatures:**
  - Ref: `mqtref.sxdg %q`
  - Value: `%q_out = mqtopt.sxdg %q_in`
- **Canonicalization:**
  - `inv sxdg → sx`
  - `sxdg %q; sxdg %q → x %q`
  - `pow(+-2) sxdg → x`
- **Matrix:** `1/2 * [1 - i, 1 + i; 1 + i, 1 - i]` (2x2 matrix)
- **Open Issues:**
  - **TODO:** Resolve global phase relationship with `rx(-π/2)`. Define power semantics.

#### 4.3.13 `rx` Gate (X-Rotation)

- **Purpose:** Rotation around the X-axis by angle θ
- **Traits:** `OneTarget`, `OneParameter`
- **Signatures:**
  - Ref: `mqtref.rx(%theta) %q`
  - Value: `%q_out = mqtopt.rx(%theta) %q_in`
- **Static variant:** `mqtref.rx(3.14159) %q`
- **Canonicalization:**
  - `rx(a) %q; rx(b) %q → rx(a + b) %q`
  - `inv rx(θ) → rx(-θ)`
  - `pow(r) rx(θ) → rx(r * θ)` for real r
- **Matrix (dynamic):** `exp(-i θ/2 X) = [cos(θ/2), -i sin(θ/2); -i sin(θ/2), cos(θ/2)]` (2x2 matrix). Static if θ constant.
- **Definition in terms of `u`:** `u(θ, -π/2, π/2) %q`

#### 4.3.14 `ry` Gate (Y-Rotation)

- **Purpose:** Rotation around the Y-axis by angle θ
- **Traits:** `OneTarget`, `OneParameter`
- **Signatures:**
  - Ref: `mqtref.ry(%theta) %q`
  - Value: `%q_out = mqtopt.ry(%theta) %q_in`
- **Static variant:** `mqtref.ry(3.14159) %q`
- **Canonicalization:**
  - `ry(a) %q; ry(b) %q → ry(a + b) %q`
  - `inv ry(θ) → ry(-θ)`
  - `pow(r) ry(θ) → ry(r * θ)` for real r
- **Matrix (dynamic):** `exp(-i θ/2 Y) = [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]` (2x2 matrix). Static if θ constant.
- **Definition in terms of `u`:** `u(θ, 0, 0) %q`

#### 4.3.15 `rz` Gate (Z-Rotation)

- **Purpose:** Rotation around the Z-axis by angle θ
- **Traits:** `OneTarget`, `OneParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.rz(%theta) %q`
  - Value: `%q_out = mqtopt.rz(%theta) %q_in`
- **Static variant:** `mqtref.rz(3.14159) %q`
- **Canonicalization:**
  - `rz(a) %q; rz(b) %q → rz(a + b) %q`
  - `inv rz(θ) → rz(-θ)`
  - `pow(r) rz(θ) → rz(r * θ)` for real r
- **Matrix (dynamic):** `exp(-i θ/2 Z) = [exp(-i θ/2), 0; 0, exp(i θ/2)]` (2x2 matrix). Static if θ constant.
- **Open Issues:**
  - **TODO:** Clarify global phase relationship: `rz(θ) == exp(iθ/2) * p(θ)`. Should canonicalization convert between these?

#### 4.3.16 `p` Gate (Phase)

- **Purpose:** Phase gate (applies a phase of θ)
- **Traits:** `OneTarget`, `OneParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.p(%theta) %q`
  - Value: `%q_out = mqtopt.p(%theta) %q_in`
- **Static variant:** `mqtref.p(3.14159) %q`
- **Canonicalization:**
  - `p(a) %q; p(b) %q → p(a + b) %q`
  - `inv p(θ) → p(-θ)`
  - `pow(r) p(θ) → p(r * θ)` for real r
- **Matrix (dynamic):** `[1, 0; 0, exp(i θ)]` (2x2 matrix). Static if θ constant.
- **Definition in terms of `u`:** `u(0, 0, θ) %q`

#### 4.3.17 `r` Gate (Arbitrary Axis Rotation)

- **Purpose:** Rotation around an arbitrary axis in the XY-plane by angles θ and φ
- **Traits:** `OneTarget`, `TwoParameter`
- **Signatures:**
  - Ref: `mqtref.r(%theta, %phi) %q`
  - Value: `%q_out = mqtopt.r(%theta, %phi) %q_in`
- **Static variant:** `mqtref.r(3.14159, 1.5708) %q`
- **Mixed variant:** `mqtref.r(%theta, 1.5708) %q`
- **Canonicalization:**
  - `inv r(θ, φ) → r(-θ, φ)`
  - `pow(real) r(θ, φ) → r(real * θ, φ)` for real `real`
  - `r(θ, 0) → rx(θ)`
  - `r(θ, π/2) → ry(θ)`
- **Matrix (dynamic):** `exp(-i θ (cos(φ) X + sin(φ) Y)) = [cos(θ/2), -i exp(-i φ) sin(θ/2); -i exp(i φ) sin(θ/2), cos(θ/2)]` (2x2 matrix). Static if θ and φ constant.
- **Definition in terms of `u`:** `u(θ, -π/2 + φ, π/2 - φ) %q`

#### 4.3.18 `u` Gate (Universal Single-Qubit)

- **Purpose:** Universal single-qubit gate (can implement any single-qubit operation)
- **Traits:** `OneTarget`, `ThreeParameter`
- **Signatures:**
  - Ref: `mqtref.u(%theta, %phi, %lambda) %q`
  - Value: `%q_out = mqtopt.u(%theta, %phi, %lambda) %q_in`
- **Static variant:** `mqtref.u(3.14159, 1.5708, 0.785398) %q`
- **Mixed variant:** `mqtref.u(%theta, 1.5708, 0.785398) %q`
- **Canonicalization:**
  - `inv u(θ, φ, λ) → u(-θ, -φ, -λ)`
  - `rx(θ) == u(θ, -π/2, π/2)`
  - `ry(θ) == u(θ, 0, 0)`
  - `p(λ) == u(0, 0, λ)`
- **Matrix (dynamic):** `p(φ) ry(θ) p(λ) = exp(i (φ + λ)/2) * rz(φ) ry(θ) rz(λ) = [cos(θ/2), -exp(i λ) sin(θ/2); exp(i φ) sin(θ/2), exp(i (φ + λ)) cos(θ/2)]` (2x2 matrix). Static if θ, φ, λ constant.

#### 4.3.19 `u2` Gate (Simplified Universal)

- **Purpose:** Simplified universal single-qubit gate (special case of `u` gate)
- **Traits:** `OneTarget`, `TwoParameter`
- \*\*Signatures
  - Ref: `mqtref.u2(%phi, %lambda) %q`
  - Value: `%q_out = mqtopt.u2(%phi, %lambda) %q_in`
- **Static variant:** `mqtref.u2(1.5708, 0.785398) %q`
- **Mixed variant:** `mqtref.u2(%phi, 0.785398) %q`
- **Canonicalization:**
  - `inv u2(φ, λ) → u2(-λ - π, -φ + π)`
  - `u2(0, π) → h`
  - `u2(0, 0) → ry(π/2)`
  - `u2(-π/2, π/2) → rx(π/2)`
- **Matrix (dynamic):** `1/sqrt(2) * [1, -exp(i λ); exp(i φ), exp(i (φ + λ))]` (2x2 matrix). Static if φ, λ constant.
- **Definition in terms of `u`:** `u2(φ, λ) == u(π/2, φ, λ)`

#### 4.3.20 `swap` Gate

- **Purpose:** Swap two qubits
- **Traits:** `TwoTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Ref: `mqtref.swap %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.swap %q0_in, %q1_in`
- **Matrix:** `[1, 0, 0, 0; 0, 0, 1, 0; 0, 1, 0, 0; 0, 0, 0, 1]` (4x4 matrix)

#### 4.3.21 `iswap` Gate

- **Purpose:** Swap two qubits with imaginary coefficient
- **Traits:** `TwoTarget`, `NoParameter`
- **Signatures:**
  - Ref: `mqtref.iswap %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.iswap %q0_in, %q1_in`
- **Canonicalization:**
  - `pow(r) iswap → xx_plus_yy(-π * r)`
- **Matrix:** `[1, 0, 0, 0; 0, 0, 1j, 0; 0, 1j, 0, 0; 0, 0, 0, 1]` (4x4 matrix)

#### 4.3.22 `dcx` Gate (Double CNOT)

- **Purpose:** Double control-NOT gate
- **Traits:** `TwoTarget`, `NoParameter`
- **Signatures:**
  - Ref: `mqtref.dcx %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.dcx %q0_in, %q1_in`
- **Canonicalization:**
  - `inv dcx %q0, q1 => dcx %q1, %q0`
- **Matrix:** `[1, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 1, 0, 0]` (4x4 matrix)

#### 4.3.23 `ecr` Gate (Echoed Cross-Resonance)

- **Purpose:** Cross-resonance gate with echo
- **Traits:** `TwoTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Ref: `mqtref.ecr %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.ecr %q0_in, %q1_in`
- **Matrix:** `1/sqrt(2) * [0, 0, 1, 1j; 0, 0, 1j, 1; 1, -1j, 0, 0; -1j, 1, 0, 0]` (4x4 matrix)

#### 4.3.24 `rxx` Gate (XX-Rotation)

- **Purpose:** General two-qubit rotation around XX.
- **Traits:** `TwoTarget`, `OneParameter`
- **Signatures:**
  - Ref: `mqtref.rxx(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.rxx(%theta) %q0_in, %q1_in`
- **Static variant:** `mqtref.rxx(3.14159) %q0, %q1`
- **Canonicalization:**
  - `inv rxx(%theta) => rxx(-%theta)`
  - `pow(r) rxx(%theta) => rxx(r * %theta)` for real r
  - `rxx(0) => remove`
  - `rxx(a) %q0, %q1; rxx(b) %q0, %q1 => rxx(a + b) %q0, %q1`
- **Matrix (dynamic):** `cos(θ/2) * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] - 1j * sin(θ/2) * [0, 0, 0, 1; 0, 0, 1, 0; 0, 1, 0, 0; 1, 0, 0, 0]` (4x4 matrix). Static if θ constant.

#### 4.3.25 `ryy` Gate (YY-Rotation)

- **Purpose:** General two-qubit gate around YY.
- **Traits:** `TwoTarget`, `OneParameter`
- **Signatures:**
  - Ref: `mqtref.ryy(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.ryy(%theta) %q0_in, %q1_in`
- **Static variant:** `mqtref.ryy(3.14159) %q0, %q1`
- **Canonicalization:**
  - `inv ryy(%theta) => ryy(-%theta)`
  - `pow(r) ryy(%theta) => ryy(r * %theta)` for real r
  - `ryy(0) => remove`
  - `ryy(a) %q0, %q1; ryy(b) %q0, %q1 => ryy(a + b) %q0, %q1`
- **Matrix (dynamic):** `cos(θ/2) * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] + 1j * sin(θ/2) * [0, 0, 0, 1; 0, 0, -1, 0; 0, -1, 0, 0; 1, 0, 0, 0]` (4x4 matrix). Static if θ constant.

#### 4.3.26 `rzx` Gate (ZX-Rotation)

- **Purpose:** General two-qubit gate around ZX.
- **Traits:** `TwoTarget`, `OneParameter`
- **Signatures:**
  - Ref: `mqtref.rzx(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.rzx(%theta) %q0_in, %q1_in`
- **Static variant:** `mqtref.rzx(3.14159) %q0, %q1`
- **Canonicalization:**
  - `inv rzx(%theta) => rzx(-%theta)`
  - `pow(r) rzx(%theta) => rzx(r * %theta)` for real r
  - `rzx(0) => remove`
  - `rzx(a) %q0, %q1; rzx(b) %q0, %q1 => rzx(a + b) %q0, %q1`
- **Matrix (dynamic):** `cos(θ/2) * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1] + 1j * sin(θ/2) * [0, -1, 0, 0; -1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0]` (4x4 matrix). Static if θ constant.

#### 4.3.27 `rzz` Gate (ZZ-Rotation)

- **Purpose:** General two-qubit gate around ZZ.
- **Traits:** `TwoTarget`, `OneParameter`, `Diagonal`
- **Signatures:**
  - Ref: `mqtref.rzz(%theta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.rzz(%theta) %q0_in, %q1_in`
- **Static variant:** `mqtref.rzz(3.14159) %q0, %q1`
- **Canonicalization:**
  - `inv rzz(%theta) => rzz(-%theta)`
  - `pow(r) rzz(%theta) => rzz(r * %theta)` for real r
  - `rzz(0) => remove`
  - `rzz(a) %q0, %q1; rzz(b) %q0, %q1 => rzz(a + b) %q0, %q1`
- **Matrix (dynamic):** `diag[exp(-i θ/2), exp(i θ/2), exp(i θ/2), exp(-i θ/2)]` (4x4 matrix). Static if θ constant.

#### 4.3.28 `xx_plus_yy` Gate

- **Purpose:** General two-qubit gate around XX+YY.
- **Traits:** `TwoTarget`, `TwoParameter`
- \*\*Signatures:
  - Ref: `mqtref.xx_plus_yy(%theta, %beta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.xx_plus_yy(%theta, %beta) %q0_in, %q1_in`
- **Static variant:** `mqtref.xx_plus_yy(3.14159, 1.5708) %q0, %q1`
- **Mixed variant:** `mqtref.xx_plus_yy(%theta, 1.5708) %q0, %q1`
- **Canonicalization:**
  - `inv xx_plus_yy(θ, β) => xx_plus_yy(-θ, β)`
  - `pow(r) xx_plus_yy(θ, β) => xx_plus_yy(r * θ, β)` for real r
  - `xx_plus_yy(θ1, β) %q0, %q1; xx_plus_yy(θ2, β) %q0, %q1 => xx_plus_yy(θ1 + θ2, β) %q0, %q1`
- **Matrix (dynamic):** `[1, 0, 0, 0; 0, cos(θ/2), sin(θ/2) * exp(-i β), 0; 0, -sin(θ/2) * exp(i β), cos(θ/2), 0; 0, 0, 0, 1]` (4x4 matrix). Static if θ and β constant.

#### 4.3.29 `xx_minus_yy` Gate

- **Purpose:** General two-qubit gate around XX-YY.
- **Traits:** `TwoTarget`, `TwoParameter`
- \*\*Signatures:
  - Ref: `mqtref.xx_minus_yy(%theta, %beta) %q0, %q1`
  - Value: `%q0_out, %q1_out = mqtopt.xx_minus_yy(%theta, %beta) %q0_in, %q1_in`
- **Static variant:** `mqtref.xx_minus_yy(3.14159, 1.5708) %q0, %q1`
- **Mixed variant:** `mqtref.xx_minus_yy(%theta, 1.5708) %q0, %q1`
- **Canonicalization:**
  - `inv xx_minus_yy(θ, β) => xx_minus_yy(-θ, β)`
  - `pow(r) xx_minus_yy(θ, β) => xx_minus_yy(r * θ, β)` for real r
  - `xx_minus_yy(θ1, β) %q0, %q1; xx_minus_yy(θ2, β) %q0, %q1 => xx_minus_yy(θ1 + θ2, β) %q0, %q1`
- **Matrix (dynamic):** `[cos(θ/2), 0, 0, -sin(θ/2) * exp(i β); 0, 1, 0, 0; 0, 0, 1, 0; sin(θ/2) * exp(-i β), 0, 0, cos(θ/2)]` (4x4 matrix). Static if θ and β constant.

#### 4.3.30 `barrier` Gate

- **Purpose:** Prevents optimization passes from reordering operations across the barrier
- **Traits:** `VariadicTarget`, `NoParameter`
- **Signatures:**
  - Ref: `mqtref.barrier %q0, %q1, ...`
  - Value: `%q0_out, %q1_out, ... = mqtopt.barrier %q0_in, %q1_in, ...`
- **Canonicalization:**
  - Barriers with no qubits can be removed
  - **TODO:** Define interaction with other optimization passes
- **Matrix:** Not applicable (compiler directive, not a unitary operation)
- **Open Issues:**
  - **TODO:** Should barrier implement `UnitaryOpInterface`? (It's effectively identity but has scheduling semantics)
  - **TODO:** Define semantics for nested barriers within modifiers

## 5. Modifier Operations

### 5.1 Overview and Philosophy

**What Are Modifiers?**

Modifiers are wrapper operations that transform or extend unitary operations without modifying the base gate definitions. They provide a composable mechanism for:

- Adding control qubits (positive or negative control)
- Inverting (taking the adjoint of) operations
- Raising operations to powers

**Key Design Principles:**

- **Single-Operation Regions:** Each modifier contains exactly one region with a single block whose only operation implements `UnitaryOpInterface`
- **Arbitrary Nesting:** Modifiers may be arbitrarily nested
- **Canonical Ordering:** Canonicalization rules flatten and reorder modifiers to a standard form: `negctrl → ctrl → pow → inv`
- **Dialect Consistency:** Both `mqtref` and `mqtopt` variants with corresponding semantics

**Value vs. Reference Semantics:**

- **Reference:** Modifiers are statements without results; wrapped operation mutates qubits in-place
- **Value:** Modifiers thread SSA values through region arguments and yield results
- **Conversion:** `mqtopt → mqtref` is straightforward; `mqtref → mqtopt` requires adding SSA values to region arguments and yields

### 5.2 Control Modifiers (`ctrl` and `negctrl`)

**Purpose:** Add additional control qubits to an operation. Control qubits can be positive (1-state control via `ctrl`) or negative (0-state control via `negctrl`).

**Signatures (shown for `ctrl`; `negctrl` is analogous):**

- **Reference:**

  ```mlir
  mqtref.ctrl(%ctrl0, %ctrl1, ...) {
    mqtref.unitaryOp %target0, %target1, ...
  }
  ```

- **Value:**
  ```mlir
  %ctrl_outs, %target_outs = mqtopt.ctrl(%ctrl_ins, %target_ins) {
    %new_targets = mqtopt.unitaryOp %target_ins
    mqtopt.yield %new_targets
  }
  ```

**Interface Implementation:**

- **Targets:** Aggregated from child unitary
- **Controls:** Control qubits from this modifier plus any controls from child unitary (flattened)
- **Parameters:** Aggregated from child unitary (none directly from modifier)
- **Static Unitary:** Available if child unitary is static

**Canonicalization:**

- Flatten nested control modifiers: `ctrl(%c1) { ctrl(%c2) { U } } → ctrl(%c1, %c2) { U }`
- Remove empty controls: `ctrl() { U } → U`
- Controlled global phase specialization: `ctrl(%c) { gphase(θ) } → p(θ) %c`
- Canonical ordering: `ctrl { negctrl { U } } → negctrl { ctrl { U } }`
- **TODO:** Define behavior when same qubit appears as both positive and negative control

**Verifiers:**

- Control and target qubits must be distinct (no qubit can be both control and target)
- All control qubits must be distinct from each other
- **TODO:** Specify verification for nested modifiers with overlapping qubit sets

**Unitary Computation:**

The unitary is computed by expanding the child operation's unitary to the larger Hilbert space defined by the additional control qubits:

```
U_controlled = |0⟩⟨0| ⊗ I_{target_space} + |1⟩⟨1| ⊗ U_{target_space}
```

For negative controls, `|1⟩⟨1|` is replaced with `|0⟩⟨0|`.

### 5.3 Inverse Modifier (`inv`)

**Purpose:** Take the adjoint (conjugate transpose) of a unitary operation.

**Signatures:**

- **Reference:**

  ```mlir
  mqtref.inv {
    mqtref.unitaryOp %targets
  }
  ```

- **Value:**
  ```mlir
  %targets_out = mqtopt.inv(%targets_in) {
    %new_targets = mqtopt.unitaryOp %targets_in
    mqtopt.yield %new_targets
  }
  ```

**Interface Implementation:**

- **Targets:** Aggregated from child unitary
- **Controls:** Aggregated from child unitary
- **Parameters:** Aggregated from child unitary (none directly from modifier)
- **Static Unitary:** Available if child unitary is static
- **Is Inverted:** Returns `true`

**Canonicalization:**

- Double inverse cancellation: `inv { inv { U } } → U`
- Hermitian gate simplification: `inv { G } → G` (for Hermitian gates)
- Parametric rotation inversion: `inv { rx(θ) } → rx(-θ)` (and similar for ry, rz, p, etc.)
- Canonical ordering: `inv { ctrl { U } } → ctrl { inv { U } }`

**Unitary Computation:**

Given unitary matrix `U`, the inverse is computed as `U† = (U̅)ᵀ` (conjugate transpose).

### 5.4 Power Modifier (`pow`)

**Purpose:** Raise a unitary operation to a given power (exponent can be integer, rational, or real).

**Signatures:**

- **Reference:**

  ```mlir
  mqtref.pow(%exponent) {
    mqtref.unitaryOp %targets
  }
  ```

  Static variant: `mqtref.pow {exponent = 2.0 : f64} { ... }`

- **Value:**
  ```mlir
  %targets_out = mqtopt.pow(%exponent, %targets_in) {
    %new_targets = mqtopt.unitaryOp %targets_in
    mqtopt.yield %new_targets
  }
  ```

**Interface Implementation:**

- **Targets:** Aggregated from child unitary
- **Controls:** Aggregated from child unitary
- **Parameters:** Aggregated from child unitary plus the exponent parameter
- **Static Unitary:** Available if child unitary and exponent are both static
- **Get Power:** Returns the exponent value

**Canonicalization:**

- Nested power flattening: `pow(a) { pow(b) { U } } → pow(a*b) { U }`
- Identity power: `pow(1) { U } → U`
- Zero power: `pow(0) { U } → remove` (becomes identity, then eliminated)
- Negative power: `pow(-r) { U } → pow(r) { inv { U } }`
- Parametric gate power folding: `pow(r) { rx(θ) } → rx(r*θ)` (for rotation gates)
- Integer power specialization: For specific gates and integer powers (e.g., `pow(2) { sx } → x`)
- Canonical ordering: `pow { ctrl { U } } → ctrl { pow { U } }`

**Unitary Computation:**

- **Integer exponents:** Matrix multiplication `U^n = U · U · ... · U` (n times)
- **Real/rational exponents:** Matrix exponentiation via eigendecomposition: `U^r = V · D^r · V†` where `U = V · D · V†` is the eigendecomposition

**Open Issues:**

- **TODO:** Define behavior for non-integer powers of non-diagonalizable gates
- **TODO:** Specify numerical precision requirements for matrix exponentiation
- **TODO:** Define semantics for complex exponents (currently not supported)

## 6. Sequence Operation (`seq`)

**Purpose:** Ordered, unnamed composition of unitary operations. Represents the application of multiple operations.

**Signatures:**

- **Reference:**

  ```mlir
  mqtref.seq {
    mqtref.h %q0
    mqtref.cx %q0, %q1
    mqtref.rz(1.57) %q1
  }
  ```

- **Value:**
  ```mlir
  %q0_out, %q1_out = mqtopt.seq(%q0_in, %q1_in) : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit) {
    %q0_1 = mqtopt.h %q0_in
    %q0_2, %q1_1 = mqtopt.cx %q0_1, %q1_in
    %q1_2 = mqtopt.rz(1.57) %q1_1
    mqtopt.yield %q0_2, %q1_2
  }
  ```

**Interface Implementation:**

- **Targets:** Aggregated from all child unitary operations (union of all targets)
- **Controls:** None (controls must be within individual operations or modifiers)
- **Parameters:** Aggregated from all child operations
- **Static Unitary:** Available if all child operations have static unitaries

**Canonicalization:**

- Empty sequence elimination: `seq { } → remove`
- Single-operation inlining: `seq { U } → U`
- Nested sequence flattening: `seq { seq { U; V }; W } → seq { U; V; W }`

**Verifiers:**

- All operations in the block must implement `UnitaryOpInterface`
- Value semantics: All yielded values must be defined within the block

**Unitary Computation:**

The composite unitary is computed as the product of child unitaries in reverse order (right-to-left multiplication, since operations apply left-to-right):

```
U_seq = U_n · U_{n-1} · ... · U_1 · U_0
```

**Conversion Between Dialects:**

- **`mqtopt → mqtref`:** Remove block arguments and results; replace argument uses with direct value references
- **`mqtref → mqtopt`:** Add block arguments for all used qubits; thread SSA values through operations; add yield with final values

## 7. User-Defined Gates & Matrix/Composite Definitions

**Purpose:** Enable users to define custom gates that can be referenced and instantiated throughout the program, similar to function definitions and calls.

### 7.1 Overview

User-defined gates provide two definition mechanisms:

1. **Matrix-based definitions:** Define a gate via its unitary matrix (efficient for small qubit counts)
2. **Sequence-based definitions:** Define a gate as a composition of existing operations (more general, better for larger circuits)

Definitions may optionally provide both representations for the same gate, which should be consistent.

**Symbol Management:**

- Gate definitions create symbols (similar to `func.func`)
- Symbols are referenced by `apply` operations (similar to `func.call`)
- Symbols have module-level scope and must be unique within a module

### 7.2 Matrix-Based Gate Definitions

**Purpose:** Define a custom gate using its unitary matrix representation.

**Signature:**

```mlir
mqt.define_matrix_gate @my_gate(%param0: f64, %param1: f64) : (2 qubits) {
  matrix = dense<[[complex_values]]> : tensor<4x4xcomplex<f64>>
  // Or symbolic expression attribute
}
```

**Open Issues:**

- **TODO:** Define complete syntax for matrix-based definitions
- **TODO:** Specify format for dynamic (parameterized) matrices using symbolic expressions
- **TODO:** Define verification rules for matrix unitarity
- **TODO:** Specify maximum practical qubit count (matrices scale as 2^n × 2^n)

### 7.3 Sequence-Based Gate Definitions

**Purpose:** Define a custom gate as a composition of existing unitary operations.

**Signature (draft):**

```mlir
mqt.define_composite_gate @my_gate(%param0: f64) : (2 qubits) {
^bb0(%q0: !mqt.qubit, %q1: !mqt.qubit):
  %q0_1 = mqt.ry(%param0) %q0
  %q0_2, %q1_1 = mqt.cx %q0_1, %q1
  mqt.return %q0_2, %q1_1
}
```

**Open Issues:**

- **TODO:** Define complete syntax with proper dialect prefix
- **TODO:** Specify parameter binding mechanism
- **TODO:** Define qubit argument conventions
- **TODO:** Clarify relationship with `seq` operation
- **TODO:** Specify whether recursive definitions are allowed

### 7.4 Gate Application (`apply`)

**Purpose:** Instantiate a user-defined gate by referencing its symbol.

**Signature (draft):**

```mlir
mqtref.apply @my_gate(%runtime_param) %q0, %q1
```

**Open Issues:**

- **TODO:** Complete specification of `apply` operation
- **TODO:** Define parameter passing mechanisms (static vs. dynamic)
- **TODO:** Specify how `apply` implements `UnitaryOpInterface`
- **TODO:** Define inlining behavior and thresholds
- **TODO:** Specify linking semantics for multi-module programs

### 7.5 Design Considerations

**Matrix vs. Sequence Trade-offs:**

- **Matrix definitions:**
  - Pros: Direct representation, efficient evaluation for small gates, exact
  - Cons: Exponential space complexity, no structure for optimization

- **Sequence definitions:**
  - Pros: Scalable, exposes structure for optimization, composable
  - Cons: May require complex unitary extraction, potential overhead

**Consistency Requirements:**

- Gates providing both matrix and sequence must be verified for consistency
- **TODO:** Define tolerance for numerical verification of consistency
- **TODO:** Specify which representation takes precedence in different contexts

## 8. Builder API

**Purpose:** Provide a programmatic API for constructing quantum programs, replacing FileCheck-based test construction with type-safe builders.

### 8.1 Design Goals

- **Ergonomic:** Easy chaining and nesting of operations
- **Type-safe:** Leverage C++ type system to catch errors at compile time
- **Dialect-aware:** Separate builders for `mqtref` and `mqtopt`
- **Testable:** Enable structural comparison of circuits in unit tests

### 8.2 Reference Semantics Builder (Draft API)

```c++
class RefQuantumProgramBuilder {
public:
  RefQuantumProgramBuilder(mlir::MLIRContext *context);

  // Initialization
  void initialize();

  // Resource management
  mlir::Value allocQubit();  // Dynamic allocation
  mlir::Value qubit(size_t index);  // Static qubit reference
  mlir::Value allocQubits(size_t count);  // Register allocation
  mlir::Value allocBits(size_t count);  // Classical register

  // Single-qubit gates (return *this for chaining)
  RefQuantumProgramBuilder& h(mlir::Value q);
  RefQuantumProgramBuilder& x(mlir::Value q);
  RefQuantumProgramBuilder& y(mlir::Value q);
  RefQuantumProgramBuilder& z(mlir::Value q);
  RefQuantumProgramBuilder& s(mlir::Value q);
  RefQuantumProgramBuilder& sdg(mlir::Value q);
  RefQuantumProgramBuilder& t(mlir::Value q);
  RefQuantumProgramBuilder& tdg(mlir::Value q);
  RefQuantumProgramBuilder& sx(mlir::Value q);
  RefQuantumProgramBuilder& sxdg(mlir::Value q);

  // Parametric single-qubit gates
  RefQuantumProgramBuilder& rx(double theta, mlir::Value q);
  RefQuantumProgramBuilder& rx(mlir::Value theta, mlir::Value q);  // Dynamic
  RefQuantumProgramBuilder& ry(double theta, mlir::Value q);
  RefQuantumProgramBuilder& ry(mlir::Value theta, mlir::Value q);
  RefQuantumProgramBuilder& rz(double theta, mlir::Value q);
  RefQuantumProgramBuilder& rz(mlir::Value theta, mlir::Value q);
  RefQuantumProgramBuilder& p(double lambda, mlir::Value q);
  RefQuantumProgramBuilder& p(mlir::Value lambda, mlir::Value q);

  // Two-qubit gates
  RefQuantumProgramBuilder& cx(mlir::Value ctrl, mlir::Value target);
  RefQuantumProgramBuilder& swap(mlir::Value q0, mlir::Value q1);
  // ... other two-qubit gates

  // Modifiers (take lambdas for body construction)
  RefQuantumProgramBuilder& ctrl(mlir::ValueRange ctrls,
                                   std::function<void(RefQuantumProgramBuilder&)> body);
  RefQuantumProgramBuilder& negctrl(mlir::ValueRange ctrls,
                                      std::function<void(RefQuantumProgramBuilder&)> body);
  RefQuantumProgramBuilder& inv(std::function<void(RefQuantumProgramBuilder&)> body);
  RefQuantumProgramBuilder& pow(double exponent,
                                 std::function<void(RefQuantumProgramBuilder&)> body);
  RefQuantumProgramBuilder& pow(mlir::Value exponent,
                                 std::function<void(RefQuantumProgramBuilder&)> body);

  // Sequence
  RefQuantumProgramBuilder& seq(std::function<void(RefQuantumProgramBuilder&)> body);

  // Gate definitions
  void defineMatrixGate(mlir::StringRef name, size_t numQubits,
                        mlir::DenseElementsAttr matrix);
  void defineCompositeGate(mlir::StringRef name, size_t numQubits,
                           mlir::ArrayRef<mlir::Type> paramTypes,
                           std::function<void(RefQuantumProgramBuilder&,
                                            mlir::ValueRange qubits,
                                            mlir::ValueRange params)> body);

  // Apply custom gate
  RefQuantumProgramBuilder& apply(mlir::StringRef gateName,
                                   mlir::ValueRange targets,
                                   mlir::ValueRange params = {});

  // Measurement and reset
  mlir::Value measure(mlir::Value q);
  RefQuantumProgramBuilder& reset(mlir::Value q);

  // Finalization
  mlir::ModuleOp finalize();
  mlir::ModuleOp getModule() const;

private:
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
  mlir::Location loc;
  // ... internal state
};
```

### 8.3 Value Semantics Builder

**TODO:** Define `OptQuantumProgramBuilder` with appropriate SSA threading semantics.

Key differences from reference builder:

- Operations return new SSA values instead of modifying in-place
- Region construction must handle argument threading
- Yield operations must be inserted appropriately

### 8.4 Usage Example

```c++
// Example: Build Bell state preparation
RefQuantumProgramBuilder builder(context);
builder.initialize();

auto q0 = builder.qubit(0);
auto q1 = builder.qubit(1);

builder.h(q0)
       .ctrl({q0}, [&](auto& b) {
           b.x(q1);
       });

auto module = builder.finalize();
```

### 8.5 C API Considerations

**TODO:** Evaluate feasibility of C API wrapper for language interoperability (Python, Rust, etc.).

Benefits:

- Enables bindings to other languages
- May eventually replace `qc::QuantumComputation` API

Challenges:

- Lambda-based modifier API needs translation
- Memory management across language boundaries

## 9. Testing Strategy

### 9.1 Philosophy

**Priority:** Structural and semantic testing > textual pattern matching

Move away from brittle FileCheck tests toward robust programmatic testing:

- **Builder-based construction:** Use builder APIs to construct circuits
- **Structural equivalence:** Compare IR structure, not SSA names
- **Interface-based verification:** Use `UnitaryOpInterface` for semantic checks
- **Numerical validation:** Compare unitary matrices with appropriate tolerances

### 9.2 Unit Testing Framework (GoogleTest)

**Test Categories:**

1. **Operation Construction:** Verify each operation constructs correctly
2. **Canonicalization:** Test each canonicalization pattern
3. **Interface Implementation:** Verify `UnitaryOpInterface` methods
4. **Matrix Extraction:** Validate static matrix computation
5. **Modifier Composition:** Test nested modifier behavior
6. **Conversion:** Test dialect conversion passes

**Example Test Structure:**

```c++
TEST(QuantumDialectTest, InverseInverseCanonicalizes) {
  MLIRContext context;
  RefQuantumProgramBuilder builder(&context);

  auto q = builder.qubit(0);
  builder.inv([&](auto& b) {
    b.inv([&](auto& b2) {
      b2.x(q);
    });
  });

  auto module = builder.finalize();

  // Apply canonicalization
  pm.addPass(createCanonicalizerPass());
  pm.run(module);

  // Verify structure: should just be single x gate
  auto func = module.lookupSymbol<func::FuncOp>("main");
  ASSERT_EQ(countOpsOfType<XOp>(func), 1);
  ASSERT_EQ(countOpsOfType<InvOp>(func), 0);
}
```

**Coverage Requirements:**

- All base gates: construction, canonicalization, matrix extraction
- All modifiers: nesting, flattening, canonical ordering
- All canonicalization rules explicitly listed in this RFC
- Negative tests for verification failures

### 9.3 Parser/Printer Round-Trip Tests (Minimal FileCheck)

**Purpose:** Ensure textual representation is stable and parseable.

**Scope:** Limited to basic round-trip validation, avoiding brittle SSA name checks.

**Example:**

```mlir
// RUN: mqt-opt %s | mqt-opt | FileCheck %s

// CHECK-LABEL: func @nested_modifiers
func.func @nested_modifiers(%q: !mqtref.qubit) {
  // CHECK: mqtref.ctrl
  // CHECK-NEXT: mqtref.pow
  // CHECK-NEXT: mqtref.inv
  // CHECK-NEXT: mqtref.x
  mqtref.ctrl %c {
    mqtref.pow {exponent = 2.0 : f64} {
      mqtref.inv {
        mqtref.x %q
      }
    }
  }
  return
}
```

**Principles:**

- Check for presence/absence of operations, not exact formatting
- Avoid checking SSA value names (they may change)
- Focus on structural properties
- Keep tests small and focused

### 9.4 Integration Tests

**TODO:** Define integration testing strategy covering:

- Multi-pass optimization pipelines
- Dialect conversion sequences
- End-to-end compilation scenarios
- Performance regression testing

### 9.5 Test Coverage Metrics

**TODO:** Establish coverage targets:

- Line coverage goal: >90%
- Branch coverage goal: >85%
- Canonicalization pattern coverage: 100%
- Interface method coverage: 100%

## 10. Implementation Roadmap

**TODO:** Define phased implementation plan with milestones.

Suggested phases:

### Phase 1: Foundation (Weeks 1-3)

- Define core type system (`mqtref.qubit`, `mqtopt.qubit`)
- Implement `UnitaryOpInterface`
- Create basic builder infrastructure
- Establish testing framework

### Phase 2: Base Gates (Weeks 4-6)

- Implement all single-qubit gates
- Implement all two-qubit gates
- Add basic canonicalization patterns
- Comprehensive unit tests for each gate

### Phase 3: Modifiers (Weeks 7-9)

- Implement `ctrl` and `negctrl`
- Implement `inv` modifier
- Implement `pow` modifier
- Test nested modifier combinations

### Phase 4: Composition (Weeks 10-12)

- Implement `seq` operation
- Add modifier flattening and ordering
- Implement gate merging canonicalization

### Phase 5: Custom Gates (Weeks 13-15)

- Design and implement gate definition operations
- Implement `apply` operation
- Add symbol table management
- Test custom gate integration

### Phase 6: Optimization (Weeks 16-18)

- Implement advanced canonicalization patterns
- Add dialect conversion passes
- Performance optimization
- Documentation

### Phase 7: Validation (Weeks 19-20)

- Comprehensive integration testing
- Performance benchmarking
- Documentation review
- Migration guide for existing code

## 11. Open Questions and Future Work

### 11.1 Critical Open Questions (Require Resolution Before Implementation)

1. **Global Phase Semantics:**
   - How should global phase gates be merged and optimized?
   - Should they implement `UnitaryOpInterface`?
   - How do they interact with circuit traversal?

2. **Global Phase Relationships:**
   - Resolve global phase differences between equivalent gates (e.g., `x` vs `rx(π)`)
   - Define whether these should be considered equivalent or distinct
   - Specify canonicalization behavior

3. **Power Semantics for Non-Diagonalizable Gates:**
   - Define behavior for fractional powers of Pauli gates
   - Specify whether `pow(1/3) x` is valid or should be rejected
   - Define numerical methods for general matrix exponentiation

4. **Custom Gate Symbol Resolution:**
   - Define complete syntax for gate definitions and applications
   - Specify linking semantics for multi-module programs
   - Define inlining heuristics

5. **Barrier Semantics:**
   - Should `barrier` implement `UnitaryOpInterface`?
   - How does it interact with modifiers?
   - Define its role in optimization passes

6. **Control Qubit Conflicts:**
   - What happens if the same qubit is both positive and negative control?
   - Should this be an error or have defined semantics?

---

## Document Metadata

- **Version:** 0.2 (Draft)
- **Last Updated:** 2025-10-06
- **Status:** Request for Comments
- **Authors:** Lukas Burgholzer (@burgholzer)
- **Reviewers:** Damian Rovara (@Drovara), Yannick Stade (@ystade), Patrick Hopf (@flowerthrower), Daniel Haag (@denialhaag), Matthias Reumann (@MatthiasReumann), Tamino Bauknecht (@taminob)
- **Target MLIR Version:** [TODO: Specify]
