# MQT MLIR Compilation Infrastructure: Technical Concept

## Executive Summary

This document describes the technical design of the MQT's MLIR-based compilation infrastructure for quantum computing.
The infrastructure provides a unified, extensible framework for quantum circuit representation, optimization, and compilation through a multi-dialect architecture built on MLIR (Multi-Level Intermediate Representation).

**Architecture Overview:**

The MQT MLIR infrastructure consists of two complementary dialects that work together to enable flexible quantum program representation and optimization:

- **Quartz Dialect** (`quartz`): Uses reference semantics with in-place qubit mutation, optimized for direct hardware mapping and straightforward translation to/from existing quantum programming languages.
- **Flux Dialect** (`flux`): Uses SSA value semantics with functional-style transformations, designed for powerful compiler optimizations and circuit transformations.

Both dialects implement a unified unitary interface, composable gate modifiers, and comprehensive canonicalization frameworks that enable seamless interoperability and progressive optimization strategies.

**Key Features:**

- **Unified Interface:** All unitary operations expose consistent APIs for introspection and composition across dialects
- **Enhanced Expressiveness:** Support for arbitrary gate modifications, custom gate definitions, and symbolic parameters
- **Optimization-Ready:** Built-in canonicalization rules and transformation hooks throughout the representation
- **Dual Semantics:** Choose the appropriate dialect for each compilation stage—reference semantics for I/O boundaries, value semantics for optimization
- **Extensible Architecture:** Designed to accommodate additional dialects and abstraction levels as the quantum computing field evolves

**Design Rationale:**

The dual-dialect architecture reflects a fundamental insight: different stages of quantum compilation benefit from different representations.
Reference semantics (Quartz) provides a natural bridge to hardware and existing quantum programming ecosystems, while value semantics (Flux) unlocks the full power of compiler analysis and optimization.
Conversion passes between dialects enable compilation strategies that leverage the strengths of each representation at the appropriate stage.

## 1. Overview and Design Philosophy

The MQT MLIR compilation infrastructure represents a comprehensive approach to quantum program compilation, building on MLIR's proven multi-level intermediate representation framework.
This design enables quantum computing to benefit from decades of classical compiler research while addressing the unique challenges of quantum circuits.

**Design Goals:**

- **Unified Unitary Interface:** Provide a coherent interface for all operations that apply or produce a unitary transformation (base gates, user-defined gates, modified operations, compositions)
- **Multi-Level Representation:** Support both reference semantics (Quartz) for hardware-oriented representations and value semantics (Flux) for optimization-oriented transformations
- **Composable Modifiers:** Enable arbitrary combinations of inversion, powering, and positive/negative control modifications through a clean modifier system
- **Custom Gate Support:** Allow users to define gates via matrix representations or compositional sequences, enabling domain-specific abstractions
- **Consistent Parameterization:** Unify handling of static and dynamic parameters with consistent ordering and introspection capabilities
- **Systematic Canonicalization:** Embed optimization rules directly at operation definitions for predictable and composable transformations
- **Normalized Representations:** Establish canonical forms (e.g., modifier ordering: `negctrl → ctrl → pow → inv`) to simplify pattern matching and optimization
- **Matrix Extraction:** Enable static matrix computation where possible while supporting symbolic composition for dynamic cases

**Architectural Principles:**

1. **Separation of Concerns:** Base gates are modifier-free; extensions are applied via wrapper operations
2. **Progressive Refinement:** Compilation proceeds through multiple passes, each operating at appropriate abstraction levels
3. **Semantic Preservation:** All transformations maintain unitary equivalence (or explicitly document approximations)
4. **Extensibility:** The architecture accommodates future dialects for pulse-level control, error correction, or domain-specific optimizations

## 2. Motivation and Context

**The MLIR Opportunity:**

MLIR provides a proven framework for building progressive compilation pipelines with multiple levels of abstraction.
By building quantum compilation infrastructure on MLIR, we gain:

- Mature ecosystem of transformation passes and analysis frameworks
- Standard infrastructure for dialect definition, conversion, and optimization
- Established patterns for SSA-based transformations and rewrites
- Interoperability with classical compilation infrastructure for hybrid quantum-classical systems

**Why Dual Dialects?**

The Quartz/Flux dual-dialect architecture reflects a key insight: quantum compilation benefits from different semantic models at different stages:

- **Quartz (Reference Semantics):** Provides an intuitive, hardware-like model where gates modify qubits in place. These semantics align naturally with:
  - Physical quantum hardware models
  - Existing quantum programming languages (OpenQASM, Qiskit, etc.)
  - Direct circuit representations
  - Backend code generation

- **Flux (Value Semantics):** Provides a functional, SSA-based model where operations produce new quantum values. These semantics enable:
  - Powerful dataflow analysis
  - Safe parallelization and reordering
  - Sophisticated optimization passes
  - Clear dependency tracking

Conversion passes between dialects allow compilation strategies to use the right representation at the right time.

## 3. Dialect Architecture

The MQT MLIR infrastructure consists of two parallel dialects with identical operation sets but different operational semantics.
This section describes the architectural design shared across both dialects.

### 3.1 Dialect Overview

**Quartz Dialect (`quartz`):**

Quartz uses **reference semantics** where quantum operations modify qubits in place, similar to how hardware physically transforms quantum states.
This model provides:

- Natural mapping to hardware execution models
- Intuitive representation for circuit descriptions
- Direct compatibility with imperative quantum programming languages
- Straightforward backend code generation

The name "Quartz" reflects the crystalline, structured nature of hardware-oriented representations—operations have fixed positions and transform states in place, like atoms in a crystal lattice.

**Example:**

```mlir
quartz.h %q              // Applies Hadamard to qubit %q in place
quartz.swap %q0, %q1       // Applies SWAP using %q0, %q1 as targets
```

**Flux Dialect (`flux`):**

Flux uses **value semantics** where quantum operations consume input qubits and produce new output values, following the functional programming and SSA paradigm.
This model enables:

- Powerful compiler optimizations through clear dataflow
- Safe reordering and parallelization analysis
- Advanced transformation passes
- Explicit dependency tracking

The name "Flux" captures the flowing, transformative nature of value-based representations—quantum states flow through operations, each transformation producing new values like a river flowing through a landscape.

**Example:**

```mlir
%q_out = flux.h %q_in                    // Consumes %q_in, produces %q_out
%q0_out, %q1_out = flux.swap %q0_in, %q1_in  // Consumes inputs, produces outputs
```

**Dialect Interoperability:**

Both dialects share operation names and core semantics, differing only in their type systems and value threading models. Bidirectional conversion passes enable flexible compilation strategies:

```
Frontend (Qiskit, OpenQASM3, ...) → Quartz → Flux → Optimizations → Flux → Quartz → Backend (QIR, OpenQASM 3, ...)
                                           ↑_____(conversion passes)_____↑
                                  ↑______________(translation passes)_____________↑
```

This architecture allows:

- Input from quantum programming languages via Quartz
- Optimization in Flux using SSA transformations
- Output to hardware backends via Quartz

**Future Extensibility:**

The architecture anticipates additional dialects for:

- Pulse-level control representations
- Error-corrected logical circuits
- Tensor network representations (e.g., ZX calculus)
- Domain-specific optimizations

Each dialect can target specific optimization goals while maintaining interoperability through conversion passes.

### 3.2 Operation Categories

All operations fall into three primary categories:

1. **Resource Operations:** Manage qubit lifetime and allocation
2. **Measurement and Reset:** Non-unitary operations that collapse or reinitialize quantum states
3. **Unitary Operations:** All operations implementing the `UnitaryOpInterface` (base gates, modifiers, sequences, custom gates)

### 3.3 Resource Operations

**Purpose:** Manage qubit lifetime and references.

**Qubit and Register Allocation:**

The MQT MLIR dialects represent quantum and classical registers using MLIR-native `memref` operations rather than custom types. This design choice offers several advantages:

- **MLIR Integration:** Seamless compatibility with existing MLIR infrastructure, enabling reuse of memory handling patterns and optimization passes
- **Implementation Efficiency:** No need to define and maintain custom register operations, significantly reducing implementation complexity
- **Enhanced Interoperability:** Easier integration with other MLIR dialects and passes, allowing for more flexible compilation pipelines
- **Sustainable Evolution:** Standard memory operations can handle transformations while allowing new features without changing the fundamental register model

**Quantum Register Representation:**

A quantum register is represented by a `memref` of type `!quartz.Qubit` or `!flux.Qubit`:

```mlir
// A quantum register with 2 qubits
%qreg = memref.alloc() : memref<2x!quartz.Qubit>

// Load qubits from the register
%q0 = memref.load %qreg[%i0] : memref<2x!quartz.Qubit>
%q1 = memref.load %qreg[%i1] : memref<2x!quartz.Qubit>
```

**Classical Register Representation:**

Classical registers follow the same pattern but use the `i1` type for Boolean measurement results:

```mlir
// A classical register with 1 bit
%creg = memref.alloc() : memref<1xi1>

// Store measurement result
%c = quartz.measure %q
memref.store %c, %creg[%i0] : memref<1xi1>
```

**Quartz Dialect (Reference Semantics):**

```mlir
%q = quartz.alloc : !quartz.qubit
quartz.dealloc %q : !quartz.qubit
%q0 = quartz.qubit 0 : !quartz.qubit  // Static qubit reference
```

**Flux Dialect (Value Semantics):**

```mlir
%q = flux.alloc : !flux.qubit
flux.dealloc %q : !flux.qubit
%q0 = flux.qubit 0 : !flux.qubit  // Static qubit reference
```

**Canonicalization Patterns:**

- Dead allocation elimination: Remove unused `alloc` operations (DCE)

### 3.4 Measurement and Reset

Non-unitary operations that do not implement the `UnitaryOpInterface`.

**Measurement Basis:** All measurements are performed in the **computational basis** (Z-basis). Measurements in other bases must be implemented by applying appropriate basis-change gates before measurement.

**Single-Qubit Measurements Only:** Multi-qubit measurements are explicitly **not supported**. Joint measurements must be decomposed into individual single-qubit measurements.

**Quartz Dialect (Reference Semantics):**

```mlir
%c = quartz.measure %q : !quartz.qubit -> i1
quartz.reset %q : !quartz.qubit
```

**Flux Dialect (Value Semantics):**

```mlir
%q_out, %c = flux.measure %q_in : !flux.qubit -> (!flux.qubit, i1)
%q_out = flux.reset %q_in : !flux.qubit -> !flux.qubit
```

**Canonicalization Patterns:**

- `reset` immediately after `alloc` → remove `reset` (already in ground state)
- Consecutive `reset` on same qubit (reference semantics) → single instance

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
- Example: `quartz.u(%theta, 1.5708, %lambda) %q` has three parameters: dynamic, static, dynamic

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
- **Consistent Signatures:** Maintain uniform syntax across Quartz and Flux dialects

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
  - Quartz: `quartz.G(param_list?) %targets : (param_types..., qubit_types...)`
  - Flux: `%out_targets = flux.G(param_list?) %in_targets : (param_types..., qubit_types...) -> (qubit_types...)`
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
  - Quartz: `quartz.gphase(%theta)`
  - Flux: `flux.gphase(%theta)`
- **Examples:**
  - Static: `quartz.gphase(3.14159)`
  - Dynamic: `quartz.gphase(%theta)`
- **Canonicalization:**
  - `gphase(0) → remove`
  - `inv(gphase(θ)) → gphase(-θ)`
  - `gphase(a); gphase(b) → gphase(a + b)` (consecutive phases merge within same scope)
  - `ctrl(%q) { gphase(θ) } → p(θ) %q` (controlled global phase becomes phase gate)
  - `negctrl(%q) { gphase(θ) } → gphase(π); p(θ) %q` (negative control specialization)
  - `pow(n) { gphase(θ) } → gphase(n*θ)`
- **Matrix:** `[exp(iθ)]` (1×1 scalar, static if θ constant)
- **Global Phase Preservation:** Global phase gates are **preserved by default** and should only be eliminated by dedicated optimization passes. They are bound to a scope (e.g., function, box) at which they are aggregated and canonicalized. Operations within the same scope may merge adjacent global phases, but global phases should not be arbitrarily removed during general canonicalization.

#### 4.3.2 `id` Gate (Identity)

- **Purpose:** Identity operation (does nothing)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`, `Diagonal`
- **Signatures:**
  - Quartz: `quartz.id %q`
  - Flux: `%q_out = flux.id %q_in`
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
  - Quartz: `quartz.x %q`
  - Flux: `%q_out = flux.x %q_in`
- **Canonicalization:**
  - `pow(1/2) x → gphase(-π/4); sx` (square root with global phase correction)
  - `pow(-1/2) x → gphase(π/4); sxdg`
  - `pow(r) x → gphase(-r*π/2); rx(r*π)` (general power translates to rotation with global phase)
- **Matrix:** `[0, 1; 1, 0]` (2x2 matrix)
- **Definition in terms of `u`:** `u(π, 0, π) %q`
- **Global Phase Relationship:** The Pauli-X gate and `rx(π)` differ by a global phase: `X = -i·exp(-iπ/2·X) = -i·rx(π)`. When raising X to fractional powers, the result is expressed as a rotation with an appropriate global phase factor.

#### 4.3.4 `y` Gate (Pauli-Y)

- **Purpose:** Pauli-Y gate (bit and phase flip)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Quartz: `quartz.y %q`
  - Flux: `%q_out = flux.y %q_in`
- **Canonicalization:**
  - `pow(r) y → gphase(-r*π/2); ry(r*π)` (general power translates to rotation with global phase)
- **Matrix:** `[0, -i; i, 0]` (2x2 matrix)
- **Definition in terms of `u`:** `u(π, π/2, π/2) %q`
- **Global Phase Relationship:** The Pauli-Y gate and `ry(π)` differ by a global phase: `Y = -i·exp(-iπ/2·Y) = -i·ry(π)`.

#### 4.3.5 `z` Gate (Pauli-Z)

- **Purpose:** Pauli-Z gate (phase flip)
- **Traits:** `OneTarget`, `NoParameter`, `Hermitian`, `Diagonal`
- **Signatures:**
  - Quartz: `quartz.z %q`
  - Flux: `%q_out = flux.z %q_in`
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
  - Quartz: `quartz.h %q`
  - Flux: `%q_out = flux.h %q_in`
- **Matrix:** `1/sqrt(2) * [1, 1; 1, -1]` (2x2 matrix)
- **Definition in terms of `u`:** `u(π/2, 0, π) %q`

#### 4.3.7 `s` Gate (S/Phase-90)

- **Purpose:** S gate (applies a phase of π/2)
- **Traits:** `OneTarget`, `NoParameter`, `Diagonal`
- **Signatures:**
  - Quartz: `quartz.s %q`
  - Flux: `%q_out = flux.s %q_in`
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
  - Quartz: `quartz.sdg %q`
  - Flux: `%q_out = flux.sdg %q_in`
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
  - Quartz: `quartz.t %q`
  - Flux: `%q_out = flux.t %q_in`
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
  - Quartz: `quartz.tdg %q`
  - Flux: `%q_out = flux.tdg %q_in`
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
  - Quartz: `quartz.sx %q`
  - Flux: `%q_out = flux.sx %q_in`
- **Canonicalization:**
  - `inv sx → sxdg`
  - `sx %q; sx %q → x %q`
  - `pow(+-2) sx → x`
  - `pow(r) sx → gphase(-r*π/4); rx(r*π/2)` (power translates to rotation with global phase)
- **Matrix:** `1/2 * [1 + i, 1 - i; 1 - i, 1 + i]` (2x2 matrix)
- **Global Phase Relationship:** `sx = exp(-iπ/4)·rx(π/2)`. Powers are handled by translating to the rotation form with appropriate global phase correction.

#### 4.3.12 `sxdg` Gate (√X-Dagger)

- **Purpose:** Square root of X-Dagger gate
- **Traits:** `OneTarget`, `NoParameter`
- **Signatures:**
  - Quartz: `quartz.sxdg %q`
  - Flux: `%q_out = flux.sxdg %q_in`
- **Canonicalization:**
  - `inv sxdg → sx`
  - `sxdg %q; sxdg %q → x %q`
  - `pow(+-2) sxdg → x`
  - `pow(r) sxdg → gphase(r*π/4); rx(-r*π/2)` (power translates to rotation with global phase)
- **Matrix:** `1/2 * [1 - i, 1 + i; 1 + i, 1 - i]` (2x2 matrix)
- **Global Phase Relationship:** `sxdg = exp(iπ/4)·rx(-π/2)`. Powers are handled by translating to the rotation form with appropriate global phase correction.

#### 4.3.13 `rx` Gate (X-Rotation)

- **Purpose:** Rotation around the X-axis by angle θ
- **Traits:** `OneTarget`, `OneParameter`
- **Signatures:**
  - Quartz: `quartz.rx(%theta) %q`
  - Flux: `%q_out = flux.rx(%theta) %q_in`
- **Static variant:** `quartz.rx(3.14159) %q`
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
  - Quartz: `quartz.ry(%theta) %q`
  - Flux: `%q_out = flux.ry(%theta) %q_in`
- **Static variant:** `quartz.ry(3.14159) %q`
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
  - Quartz: `quartz.rz(%theta) %q`
  - Flux: `%q_out = flux.rz(%theta) %q_in`
- **Static variant:** `quartz.rz(3.14159) %q`
- **Canonicalization:**
  - `rz(a) %q; rz(b) %q → rz(a + b) %q`
  - `inv rz(θ) → rz(-θ)`
  - `pow(r) rz(θ) → rz(r * θ)` for real r
- **Matrix (dynamic):** `exp(-i θ/2 Z) = [exp(-i θ/2), 0; 0, exp(i θ/2)]` (2x2 matrix). Static if θ constant.
- **Relationship with `p` gate:** The `rz` and `p` gates differ by a global phase: `rz(θ) = exp(iθ/2)·p(θ)`. However, **`rz` and `p` should remain separate** and are **not canonicalized** into each other, as they represent different conventions that may be important for hardware backends or algorithm implementations.

#### 4.3.16 `p` Gate (Phase)

- **Purpose:** Phase gate (applies a phase of θ)
- **Traits:** `OneTarget`, `OneParameter`, `Diagonal`
- **Signatures:**
  - Quartz: `quartz.p(%theta) %q`
  - Flux: `%q_out = flux.p(%theta) %q_in`
- **Static variant:** `quartz.p(3.14159) %q`
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
  - Quartz: `quartz.r(%theta, %phi) %q`
  - Flux: `%q_out = flux.r(%theta, %phi) %q_in`
- **Static variant:** `quartz.r(3.14159, 1.5708) %q`
- **Mixed variant:** `quartz.r(%theta, 1.5708) %q`
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
  - Quartz: `quartz.u(%theta, %phi, %lambda) %q`
  - Flux: `%q_out = flux.u(%theta, %phi, %lambda) %q_in`
- **Static variant:** `quartz.u(3.14159, 1.5708, 0.785398) %q`
- **Mixed variant:** `quartz.u(%theta, 1.5708, 0.785398) %q`
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
  - Quartz: `quartz.u2(%phi, %lambda) %q`
  - Flux: `%q_out = flux.u2(%phi, %lambda) %q_in`
- **Static variant:** `quartz.u2(1.5708, 0.785398) %q`
- **Mixed variant:** `quartz.u2(%phi, 0.785398) %q`
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
  - Quartz: `quartz.swap %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.swap %q0_in, %q1_in`
- **Matrix:** `[1, 0, 0, 0; 0, 0, 1, 0; 0, 1, 0, 0; 0, 0, 0, 1]` (4x4 matrix)

#### 4.3.21 `iswap` Gate

- **Purpose:** Swap two qubits with imaginary coefficient
- **Traits:** `TwoTarget`, `NoParameter`
- **Signatures:**
  - Quartz: `quartz.iswap %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.iswap %q0_in, %q1_in`
- **Canonicalization:**
  - `pow(r) iswap → xx_plus_yy(-π * r)`
- **Matrix:** `[1, 0, 0, 0; 0, 0, 1j, 0; 0, 1j, 0, 0; 0, 0, 0, 1]` (4x4 matrix)

#### 4.3.22 `dcx` Gate (Double CNOT)

- **Purpose:** Double control-NOT gate
- **Traits:** `TwoTarget`, `NoParameter`
- **Signatures:**
  - Quartz: `quartz.dcx %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.dcx %q0_in, %q1_in`
- **Canonicalization:**
  - `inv dcx %q0, q1 => dcx %q1, %q0`
- **Matrix:** `[1, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 1, 0, 0]` (4x4 matrix)

#### 4.3.23 `ecr` Gate (Echoed Cross-Resonance)

- **Purpose:** Cross-resonance gate with echo
- **Traits:** `TwoTarget`, `NoParameter`, `Hermitian`
- **Signatures:**
  - Quartz: `quartz.ecr %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.ecr %q0_in, %q1_in`
- **Matrix:** `1/sqrt(2) * [0, 0, 1, 1j; 0, 0, 1j, 1; 1, -1j, 0, 0; -1j, 1, 0, 0]` (4x4 matrix)

#### 4.3.24 `rxx` Gate (XX-Rotation)

- **Purpose:** General two-qubit rotation around XX.
- **Traits:** `TwoTarget`, `OneParameter`
- **Signatures:**
  - Quartz: `quartz.rxx(%theta) %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.rxx(%theta) %q0_in, %q1_in`
- **Static variant:** `quartz.rxx(3.14159) %q0, %q1`
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
  - Quartz: `quartz.ryy(%theta) %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.ryy(%theta) %q0_in, %q1_in`
- **Static variant:** `quartz.ryy(3.14159) %q0, %q1`
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
  - Quartz: `quartz.rzx(%theta) %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.rzx(%theta) %q0_in, %q1_in`
- **Static variant:** `quartz.rzx(3.14159) %q0, %q1`
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
  - Quartz: `quartz.rzz(%theta) %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.rzz(%theta) %q0_in, %q1_in`
- **Static variant:** `quartz.rzz(3.14159) %q0, %q1`
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
  - Quartz: `quartz.xx_plus_yy(%theta, %beta) %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.xx_plus_yy(%theta, %beta) %q0_in, %q1_in`
- **Static variant:** `quartz.xx_plus_yy(3.14159, 1.5708) %q0, %q1`
- **Mixed variant:** `quartz.xx_plus_yy(%theta, 1.5708) %q0, %q1`
- **Canonicalization:**
  - `inv xx_plus_yy(θ, β) => xx_plus_yy(-θ, β)`
  - `pow(r) xx_plus_yy(θ, β) => xx_plus_yy(r * θ, β)` for real r
  - `xx_plus_yy(θ1, β) %q0, %q1; xx_plus_yy(θ2, β) %q0, %q1 => xx_plus_yy(θ1 + θ2, β) %q0, %q1`
- **Matrix (dynamic):** `[1, 0, 0, 0; 0, cos(θ/2), sin(θ/2) * exp(-i β), 0; 0, -sin(θ/2) * exp(i β), cos(θ/2), 0; 0, 0, 0, 1]` (4x4 matrix). Static if θ and β constant.

#### 4.3.29 `xx_minus_yy` Gate

- **Purpose:** General two-qubit gate around XX-YY.
- **Traits:** `TwoTarget`, `TwoParameter`
- \*\*Signatures:
  - Quartz: `quartz.xx_minus_yy(%theta, %beta) %q0, %q1`
  - Flux: `%q0_out, %q1_out = flux.xx_minus_yy(%theta, %beta) %q0_in, %q1_in`
- **Static variant:** `quartz.xx_minus_yy(3.14159, 1.5708) %q0, %q1`
- **Mixed variant:** `quartz.xx_minus_yy(%theta, 1.5708) %q0, %q1`
- **Canonicalization:**
  - `inv xx_minus_yy(θ, β) => xx_minus_yy(-θ, β)`
  - `pow(r) xx_minus_yy(θ, β) => xx_minus_yy(r * θ, β)` for real r
  - `xx_minus_yy(θ1, β) %q0, %q1; xx_minus_yy(θ2, β) %q0, %q1 => xx_minus_yy(θ1 + θ2, β) %q0, %q1`
- **Matrix (dynamic):** `[cos(θ/2), 0, 0, -sin(θ/2) * exp(i β); 0, 1, 0, 0; 0, 0, 1, 0; sin(θ/2) * exp(-i β), 0, 0, cos(θ/2)]` (4x4 matrix). Static if θ and β constant.

#### 4.3.30 `barrier` Gate

- **Purpose:** Prevents optimization passes from reordering operations across the barrier
- **Traits:** `OneTarget`, `TwoTarget`, `NoParameter` (overloaded for different qubit counts)
- **Signatures:**
  - Quartz: `quartz.barrier %q0, %q1, ...`
  - Flux: `%q0_out, %q1_out, ... = flux.barrier %q0_in, %q1_in, ...`
- **Semantics:** The `barrier` operation implements the `UnitaryOpInterface` and is treated similarly to the identity gate from a unitary perspective. However, it serves as a compiler directive that constrains optimization: operations cannot be moved across a barrier boundary.
- **Canonicalization:**
  - Barriers with no qubits can be removed
  - `barrier; barrier` on same qubits → single `barrier` (adjacent barriers merge)
  - `inv { barrier } → barrier` (barrier is self-adjoint)
  - `pow(r) { barrier } → barrier` (any power of barrier is still barrier)
- **Matrix:** Identity matrix of appropriate dimension (2^n × 2^n for n qubits)
- **UnitaryOpInterface Implementation:** Returns identity matrix, no parameters, no controls, targets are the specified qubits

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
- **Dialect Consistency:** Both `quartz` and `flux` variants with corresponding semantics

**Value vs. Reference Semantics:**

- **Quartz (Reference):** Modifiers are statements without results; wrapped operation mutates qubits in-place
- **Flux (Value):** Modifiers thread SSA values through region arguments and yield results
- **Conversion:** `flux → quartz` is straightforward; `quartz → flux` requires adding SSA values to region arguments and yields

### 5.2 Control Modifiers (`ctrl` and `negctrl`)

**Purpose:** Add additional control qubits to an operation. Control qubits can be positive (1-state control via `ctrl`) or negative (0-state control via `negctrl`).

**Signatures (shown for `ctrl`; `negctrl` is analogous):**

- **Quartz (Reference Semantics):**

  ```mlir
  quartz.ctrl(%ctrl0, %ctrl1, ...) {
    quartz.unitaryOp %target0, %target1, ...
  }
  ```

- **Flux (Value Semantics):**
  ```mlir
  %ctrl_outs, %target_outs = flux.ctrl(%ctrl_ins, %target_ins) {
    %new_targets = flux.unitaryOp %target_ins
    flux.yield %new_targets
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
- **Control Modifier Exclusivity:** A qubit must **not** appear in both positive (`ctrl`) and negative (`negctrl`) control modifiers. The sets of positive and negative controls must be **disjoint** with no overlap. If a qubit appears in both, the operation is invalid and must be rejected during verification.

**Unitary Computation:**

The unitary is computed by expanding the child operation's unitary to the larger Hilbert space defined by the additional control qubits:

```
U_controlled = |0⟩⟨0| ⊗ I_{target_space} + |1⟩⟨1| ⊗ U_{target_space}
```

For negative controls, `|1⟩⟨1|` is replaced with `|0⟩⟨0|`.

### 5.3 Inverse Modifier (`inv`)

**Purpose:** Take the adjoint (conjugate transpose) of a unitary operation.

**Signatures:**

- **Quartz (Reference Semantics):**

  ```mlir
  quartz.inv {
    quartz.unitaryOp %targets
  }
  ```

- **Flux (Value Semantics):**
  ```mlir
  %targets_out = flux.inv(%targets_in) {
    %new_targets = flux.unitaryOp %targets_in
    flux.yield %new_targets
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

- **Quartz (Reference Semantics):**

  ```mlir
  quartz.pow(%exponent) {
    quartz.unitaryOp %targets
  }
  ```

  Static variant: `quartz.pow {exponent = 2.0 : f64} { ... }`

- **Flux (Value Semantics):**
  ```mlir
  %targets_out = flux.pow(%exponent, %targets_in) {
    %new_targets = flux.unitaryOp %targets_in
    flux.yield %new_targets
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

**Well-Definedness:** Since all quantum gates are unitary matrices, they are always diagonalizable (unitaries are normal operators). Therefore, **non-integer powers are always well-defined** through eigendecomposition, regardless of the specific gate.

**Numerical Precision:** Matrix exponentiation should be computed to **machine precision** (typically double precision floating point, ~15-16 decimal digits). Implementations should use numerically stable algorithms for eigendecomposition.

**Complex Exponents:** Complex exponents are **not supported** and will **not be supported** in the future, as they do not have meaningful physical interpretations for quantum gates. Only real-valued exponents are permitted.

## 6. Box Operation (`box`)

**Purpose:** Scoped composition of unitary operations with timing and optimization constraints. Inspired by OpenQASM 3.0's `box` statement, this operation encapsulates a sequence of operations while constraining how optimizations may interact with them.

**Signatures:**

- **Quartz (Reference Semantics):**

  ```mlir
  quartz.box {
    quartz.h %q0
    quartz.cx %q0, %q1
    quartz.rz(1.57) %q1
  }
  ```

- **Flux (Value Semantics):**
  ```mlir
  %q0_out, %q1_out = flux.box(%q0_in, %q1_in) : (!flux.qubit, !flux.qubit) -> (!flux.qubit, !flux.qubit) {
    %q0_1 = flux.h %q0_in
    %q0_2, %q1_1 = flux.cx %q0_1, %q1_in
    %q1_2 = flux.rz(1.57) %q1_1
    flux.yield %q0_2, %q1_2
  }
  ```

**Interface Implementation:**

- **Targets:** Aggregated from all child unitary operations (union of all targets)
- **Controls:** None (controls must be within individual operations or modifiers)
- **Parameters:** Aggregated from all child operations
- **Static Unitary:** Available if all child operations have static unitaries

**Canonicalization:**

- Empty sequence elimination: `box { } → remove`
- Single-operation inlining: `box { U } → U`
- Nested sequence flattening: `box { box { U; V }; W } → box { U; V; W }`

**Verifiers:**

- All operations in the block must implement `UnitaryOpInterface`
- Value semantics: All yielded values must be defined within the block

**Unitary Computation:**

The composite unitary is computed as the product of child unitaries in reverse order (right-to-left multiplication, since operations apply left-to-right):

```
U_box = U_n · U_{n-1} · ... · U_1 · U_0
```

**Conversion Between Dialects:**

- **`flux → quartz`:** Remove block arguments and results; replace argument uses with direct value references
- **`quartz → flux`:** Add block arguments for all used qubits; thread SSA values through operations; add yield with final values

## 8. Builder API

**Purpose:** Provide a programmatic API for constructing quantum programs within the MQT MLIR infrastructure. The builder APIs offer type-safe, ergonomic interfaces for both Quartz and Flux dialects.

### 8.1 Design Goals

- **Ergonomic:** Easy chaining and nesting of operations
- **Type-safe:** Leverage C++ type system to catch errors at compile time
- **Dialect-aware:** Separate builders for Quartz and Flux
- **Testable:** Enable structural comparison of circuits in unit tests

### 8.2 Quartz Builder (Reference Semantics)

```c++
class QuartzProgramBuilder {
public:
  QuartzProgramBuilder(mlir::MLIRContext *context);

  // Initialization
  void initialize();

  // Resource management
  mlir::Value allocQubit();  // Dynamic allocation
  mlir::Value qubit(size_t index);  // Static qubit reference
  mlir::Value allocQubits(size_t count);  // Register allocation (returns memref)
  mlir::Value allocBits(size_t count);  // Classical register (returns memref)

  // Single-qubit gates (return *this for chaining)
  QuartzProgramBuilder& h(mlir::Value q);
  QuartzProgramBuilder& x(mlir::Value q);
  QuartzProgramBuilder& y(mlir::Value q);
  QuartzProgramBuilder& z(mlir::Value q);
  QuartzProgramBuilder& s(mlir::Value q);
  QuartzProgramBuilder& sdg(mlir::Value q);
  QuartzProgramBuilder& t(mlir::Value q);
  QuartzProgramBuilder& tdg(mlir::Value q);
  QuartzProgramBuilder& sx(mlir::Value q);
  QuartzProgramBuilder& sxdg(mlir::Value q);

  // Parametric single-qubit gates
  QuartzProgramBuilder& rx(double theta, mlir::Value q);
  QuartzProgramBuilder& rx(mlir::Value theta, mlir::Value q);  // Dynamic
  QuartzProgramBuilder& ry(double theta, mlir::Value q);
  QuartzProgramBuilder& ry(mlir::Value theta, mlir::Value q);
  QuartzProgramBuilder& rz(double theta, mlir::Value q);
  QuartzProgramBuilder& rz(mlir::Value theta, mlir::Value q);
  QuartzProgramBuilder& p(double lambda, mlir::Value q);
  QuartzProgramBuilder& p(mlir::Value lambda, mlir::Value q);

  // Two-qubit gates
  QuartzProgramBuilder& swap(mlir::Value q0, mlir::Value q1);
  QuartzProgramBuilder& iswap(mlir::Value q0, mlir::Value q1);
  QuartzProgramBuilder& dcx(mlir::Value q0, mlir::Value q1);
  QuartzProgramBuilder& ecr(mlir::Value q0, mlir::Value q1);
  // ... other two-qubit gates

  // Convenience gates (inspired by qc::QuantumComputation API)
  // Standard controlled gates
  QuartzProgramBuilder& cx(mlir::Value ctrl, mlir::Value target);  // CNOT
  QuartzProgramBuilder& cy(mlir::Value ctrl, mlir::Value target);
  QuartzProgramBuilder& cz(mlir::Value ctrl, mlir::Value target);
  QuartzProgramBuilder& ch(mlir::Value ctrl, mlir::Value target);
  QuartzProgramBuilder& crx(double theta, mlir::Value ctrl, mlir::Value target);
  QuartzProgramBuilder& cry(double theta, mlir::Value ctrl, mlir::Value target);
  QuartzProgramBuilder& crz(double theta, mlir::Value ctrl, mlir::Value target);

  // Multi-controlled gates (arbitrary number of controls)
  QuartzProgramBuilder& mcx(mlir::ValueRange ctrls, mlir::Value target);  // Toffoli, etc.
  QuartzProgramBuilder& mcy(mlir::ValueRange ctrls, mlir::Value target);
  QuartzProgramBuilder& mcz(mlir::ValueRange ctrls, mlir::Value target);
  QuartzProgramBuilder& mch(mlir::ValueRange ctrls, mlir::Value target);

  // Modifiers (take lambdas for body construction)
  QuartzProgramBuilder& ctrl(mlir::ValueRange ctrls,
                                   std::function<void(QuartzProgramBuilder&)> body);
  QuartzProgramBuilder& negctrl(mlir::ValueRange ctrls,
                                      std::function<void(QuartzProgramBuilder&)> body);
  QuartzProgramBuilder& inv(std::function<void(QuartzProgramBuilder&)> body);
  QuartzProgramBuilder& pow(double exponent,
                                 std::function<void(QuartzProgramBuilder&)> body);
  QuartzProgramBuilder& pow(mlir::Value exponent,
                                 std::function<void(QuartzProgramBuilder&)> body);

  // Box (scoped sequence with optimization constraints)
  QuartzProgramBuilder& box(std::function<void(QuartzProgramBuilder&)> body);

  // Gate definitions
  void defineMatrixGate(mlir::StringRef name, size_t numQubits,
                        mlir::DenseElementsAttr matrix);
  void defineCompositeGate(mlir::StringRef name, size_t numQubits,
                           mlir::ArrayRef<mlir::Type> paramTypes,
                           std::function<void(QuartzProgramBuilder&,
                                            mlir::ValueRange qubits,
                                            mlir::ValueRange params)> body);

  // Apply custom gate
  QuartzProgramBuilder& apply(mlir::StringRef gateName,
                                   mlir::ValueRange targets,
                                   mlir::ValueRange params = {});

  // Measurement and reset
  mlir::Value measure(mlir::Value q);
  QuartzProgramBuilder& reset(mlir::Value q);

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

### 8.3 Flux Builder (Value Semantics)

The `FluxProgramBuilder` follows the same design principles as the Quartz builder but with SSA value threading:

**Key Differences:**

- **Return Values:** All gate operations return new SSA values representing the output qubits
- **Threading:** Operations consume input qubits and produce output qubits
- **Region Construction:** Modifiers and box operations handle proper argument threading and yield insertion
- **Type Signatures:** Uses `!flux.qubit` instead of `!quartz.qubit`

**Example API Sketch:**

```c++
class FluxProgramBuilder {
public:
  FluxProgramBuilder(mlir::MLIRContext *context);

  // Single-qubit gates return new qubit values
  mlir::Value h(mlir::Value q_in);
  mlir::Value x(mlir::Value q_in);
  mlir::Value rx(double theta, mlir::Value q_in);

  // Two-qubit gates return tuple of output qubits
  std::pair<mlir::Value, mlir::Value> cx(mlir::Value ctrl_in, mlir::Value target_in);
  std::pair<mlir::Value, mlir::Value> swap(mlir::Value q0_in, mlir::Value q1_in);

  // Convenience multi-controlled gates
  mlir::Value mcx(mlir::ValueRange ctrl_ins, mlir::Value target_in,
                   mlir::ValueRange& ctrl_outs);

  // Modifiers handle value threading automatically
  mlir::ValueRange ctrl(mlir::ValueRange ctrl_ins, mlir::ValueRange target_ins,
                         std::function<mlir::ValueRange(FluxProgramBuilder&,
                                                        mlir::ValueRange)> body);

  // ... similar methods to QuartzProgramBuilder
};
```

### 8.4 Usage Examples

```c++
// Example: Build Bell state preparation with Quartz dialect
QuartzProgramBuilder builder(context);
builder.initialize();

auto q0 = builder.qubit(0);
auto q1 = builder.qubit(1);

// Using convenience method
builder.h(q0).cx(q0, q1);

// Equivalent using modifier
builder.h(q0)
       .ctrl({q0}, [&](auto& b) {
           b.x(q1);
       });

auto module = builder.finalize();
```

```c++
// Example: Build Toffoli gate with multi-controlled convenience
QuartzProgramBuilder builder(context);
builder.initialize();

auto q0 = builder.qubit(0);
auto q1 = builder.qubit(1);
auto q2 = builder.qubit(2);

// Toffoli using convenience method
builder.mcx({q0, q1}, q2);

// Equivalent using modifier
builder.ctrl({q0, q1}, [&](auto& b) {
    b.x(q2);
});
```

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
func.func @nested_modifiers(%q: !quartz.qubit) {
  // CHECK: quartz.ctrl
  // CHECK-NEXT: quartz.pow
  // CHECK-NEXT: quartz.inv
  // CHECK-NEXT: quartz.x
  quartz.ctrl %c {
    quartz.pow {exponent = 2.0 : f64} {
      quartz.inv {
        quartz.x %q
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

Integration tests validate the complete MQT MLIR compilation infrastructure, covering end-to-end scenarios from frontend ingestion through optimization to backend emission.

**Test Coverage:**

1. **Frontend Translation:**
   - **Qiskit → MLIR:** Convert Qiskit `QuantumCircuit` objects to Quartz/Flux dialects
   - **OpenQASM 3.0 → MLIR:** Parse OpenQASM 3.0 source and lower to MLIR
   - **`qc::QuantumComputation` → MLIR:** Translate from existing MQT Core IR to MLIR representation

2. **Compiler Pipeline Integration:**
   - Execute default optimization pass pipeline on translated circuits
   - Verify correctness through unitary equivalence checks
   - Test dialect conversions (Quartz ↔ Flux) within the pipeline
   - Validate canonicalization and optimization passes

3. **Backend Translation:**
   - **MLIR → QIR:** Lower optimized MLIR to Quantum Intermediate Representation (QIR)
   - Verify QIR output is valid and executable
   - Round-trip testing where applicable

4. **End-to-End Scenarios:**
   - Common quantum algorithms (Bell state, GHZ, QPE, VQE ansätze)
   - Parameterized circuits with classical optimization loops
   - Circuits with measurements and classical control flow
   - Large-scale circuits (100+ qubits, 1000+ gates)

## 10. Implementation Roadmap

**Timeline:** 14-week implementation plan to complete the MQT MLIR compilation infrastructure.

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Establish core infrastructure

- **Week 1:**
  - Define core type system (`quartz.qubit`, `flux.qubit`)
  - Implement `UnitaryOpInterface` definition
  - Set up basic CMake integration and build system
  - Establish GoogleTest framework for unit testing

- **Week 2:**
  - Create basic builder infrastructure (QuartzProgramBuilder skeleton)
  - Implement resource operations (alloc, dealloc, qubit reference)
  - Add measurement and reset operations
  - Basic parser/printer for qubit types

**Deliverable:** Compiling infrastructure with basic qubit operations

### Phase 2: Base Gates - Core Set (Weeks 2-3)

**Goal:** Implement essential gates for basic circuits

- **Week 2-3:**
  - Implement Pauli gates (x, y, z, id)
  - Implement Hadamard (h)
  - Implement phase gates (s, sdg, t, tdg, p)
  - Implement rotation gates (rx, ry, rz)
  - Implement universal gates (u, u2)
  - Implement convenience gates (sx, sxdg)
  - Basic canonicalization patterns for each gate
  - Unit tests for all implemented gates

**Deliverable:** Core single-qubit gate set with tests

### Phase 3: Base Gates - Two-Qubit Set (Week 4)

**Goal:** Complete two-qubit gate library

- Implement swap, iswap, dcx, ecr
- Implement parametric two-qubit gates (rxx, ryy, rzz, rzx, xx_plus_yy, xx_minus_yy)
- Implement barrier operation
- Implement global phase gate (gphase)
- Advanced canonicalization patterns
- Comprehensive unit tests

**Deliverable:** Complete base gate set

### Phase 4: Modifiers (Week 5)

**Goal:** Implement composable modifier system

- Implement `ctrl` modifier (positive controls)
- Implement `negctrl` modifier (negative controls)
- Implement `inv` modifier
- Implement `pow` modifier
- Nested modifier verification and canonicalization
- Canonical ordering implementation (negctrl → ctrl → pow → inv)
- Modifier flattening and optimization
- Extensive unit tests for modifier combinations

**Deliverable:** Working modifier system with canonicalization

### Phase 5: Composition & Box (Week 6)

**Goal:** Enable circuit composition

- Implement `box` operation (replaces `seq`)
- Add box-level canonicalization
- Implement optimization constraint enforcement
- Builder API for box operations
- Integration with modifiers
- Unit tests for box semantics

**Deliverable:** Box operation with proper scoping

### Phase 6: Custom Gates (Week 7)

**Goal:** User-defined gate support

- Design and implement gate definition operations (matrix-based)
- Design and implement gate definition operations (sequence-based)
- Implement `apply` operation
- Symbol table management and resolution
- Consistency verification for dual representations
- Inlining infrastructure
- Unit tests for custom gates

**Deliverable:** Custom gate definition and application

### Phase 7: Builder API & Convenience (Week 8)

**Goal:** Ergonomic programmatic construction

- Complete RefQuantumProgramBuilder implementation
- Complete OptQuantumProgramBuilder implementation
- Add convenience methods (cx, cz, ccx, mcx, mcy, mcz, mch, etc.)
- Builder unit tests
- Example programs using builders

**Deliverable:** Production-ready builder APIs

### Phase 8: Dialect Conversion (Week 9)

**Goal:** Enable transformations between dialects

- Implement MQTRef → MQTOpt conversion pass
- Implement MQTOpt → MQTRef conversion pass
- Handle modifier and box conversions
- SSA value threading logic
- Conversion unit tests

**Deliverable:** Working dialect conversion infrastructure

### Phase 9: Frontend Integration (Week 10)

**Goal:** Connect to existing quantum software

- Qiskit → MLIR translation
- OpenQASM 3.0 → MLIR translation
- `qc::QuantumComputation` → MLIR translation
- Parser infrastructure for OpenQASM
- Integration tests for each frontend

**Deliverable:** Multiple input format support

### Phase 10: Optimization Passes (Week 11)

**Goal:** Implement optimization infrastructure

- Advanced canonicalization patterns
- Gate fusion passes
- Circuit optimization passes
- Dead code elimination
- Constant folding and propagation
- Pass pipeline configuration
- Performance benchmarks

**Deliverable:** Optimization pass suite

### Phase 11: Backend Integration (Week 12)

**Goal:** QIR lowering

- MLIR → QIR lowering pass
- QIR emission and verification
- End-to-end integration tests (Qiskit → MLIR → QIR)
- Documentation for QIR mapping

**Deliverable:** Complete compilation pipeline

### Phase 12: Testing & Documentation (Week 13)

**Goal:** Production readiness

- Comprehensive integration test suite
- Parser/printer round-trip tests
- Performance regression tests
- API documentation
- Tutorial examples
- Migration guide for existing code

**Deliverable:** Production-ready system with documentation

### Phase 13: Polish & Release (Week 14)

**Goal:** Release preparation

- Bug fixes from testing
- Performance optimization
- Code review and cleanup
- Release notes
- Final documentation review
- Announce and deploy

**Deliverable:** Released dialect system

**Parallelization Strategy:**

- Base gate implementation (Phases 2-3) can partially overlap
- Builder API (Phase 7) can be developed in parallel with custom gates (Phase 6)
- Frontend integration (Phase 9) can start once base operations are stable (after Phase 5)
- Testing and documentation (Phase 12) should be ongoing throughout

**Risk Mitigation:**

- **Critical Path:** Phases 1-4 are sequential and critical
- **Buffer Time:** Phase 13 provides 1-week buffer for unforeseen issues
- **Early Testing:** Integration tests should begin in Phase 9 to catch issues early
- **Incremental Delivery:** Each phase produces a working subset to enable testing
