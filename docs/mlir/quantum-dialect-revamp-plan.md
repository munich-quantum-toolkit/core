## 1. Overview and Goals

### 1.1 Current State

We have two quantum dialects:

- `mqtref`: Reference semantics (side-effect based)
- `mqtopt`: Value semantics (SSA based)

Both support basic gates and quantum operations but suffer from:

- **Verbose builders and cumbersome usage**: Creating controlled gates requires lengthy builder calls
- **Gate modifiers embedded inconsistently**: Some gates have control operands, others use attributes, while others do not even exist yet, creating analysis complexity
- **Inconsistent and incomplete interfaces across dialects**: Different ways to query the same logical information
- **Limited composability**: No clean way to compose modifiers (e.g., controlled inverse gates)

### 1.2 Proposed Changes

This plan proposes a fundamental redesign that addresses these issues through:

1. **Compositional Design**: Modifiers become explicit wrapper operations that can be nested and combined
2. **Unified Interface**: Single `UnitaryOpInterface` for all quantum operations enabling uniform analysis
3. **Improved Ergonomics**: Builder APIs and parser sugar for common patterns without sacrificing expressiveness
4. **Shared Infrastructure**: Common traits, interfaces, and utilities to reduce code duplication

**Rationale**: By making modifiers compositional, we get exponential expressiveness with linear implementation cost. It is also fairly close to how established languages like OpenQASM or Qiskit work, which makes it easier to transition to the new dialect.

## 2. Architecture Overview

### 2.1 Dialect Structure

```
Common/
├── Interfaces (UnitaryOpInterface, etc.)
├── Traits (Hermitian, Diagonal, SingleTarget, etc.)
└── Support (UnitaryExpr library)

mqtref/
├── Types (Qubit)
├── Base Gates (X, RX, CNOT, etc.)
├── Modifiers (ctrl, inv, pow)
├── Sequences (seq)
└── Resources (alloc, measure, etc.)

mqtopt/
└── (Same structure with value semantics)
```

### 2.2 Key Design Principles

1. **Base gates are minimal**: No control operands or modifier attributes - each gate does exactly one thing
2. **Modifiers wrap operations**: Controls, inverses, and powers become explicit wrapper operations with regions
3. **Uniform interface**: Single way to query any quantum operation regardless of dialect or complexity
4. **Dialect-specific semantics**: Reference vs value threading handled appropriately by each dialect

**Rationale**: This separation of concerns makes analysis passes simpler - they can focus on the mathematical structure without worrying about the myriad ways modifiers might be encoded. It also makes the IR more predictable and easier to canonicalize.

## 3. Types and Memory Model

### 3.1 Quantum Types

Both dialects use a single `Qubit` type with different semantics:

- `!mqtref.qubit` (reference semantics): Represents a stateful quantum register location
- `!mqtopt.qubit` (value semantics): Represents an SSA value carrying quantum information

**Rationale**: While conceptually similar, the semantic difference is crucial - mqtref qubits can be mutated in place, while mqtopt qubits follow single-static-assignment and must be "threaded" through operations.

### 3.2 Register Handling

Use standard MLIR `memref`s for quantum and classical registers:

```mlir
// Quantum register allocation and access
%qreg = memref.alloc() : memref<2x!mqtref.qubit>
%q0 = memref.load %qreg[%c0] : memref<2x!mqtref.qubit>

// Classical register for measurements
%creg = memref.alloc() : memref<2xi1>
%bit = mqtref.measure %q0 : i1
memref.store %bit, %creg[%c0] : memref<2xi1>
```

**Rationale**: Using standard MLIR constructs (memref) instead of custom quantum register types allows us to leverage existing MLIR analyses and transformations. This also provides a clear separation between the quantum computational model and classical memory management.

## 4. Base Gate Operations

### 4.1 Design Philosophy

Base gates are the atomic quantum operations and contain only:

- **Target operands**: Fixed arity determined by gate type (via traits)
- **Parameters**: Rotation angles, phases, etc. (both static and dynamic)
- **No control operands**: Controls are handled by wrapper operations
- **No modifier attributes**: Inverses and powers are handled by wrapper operations

**Rationale**: This minimal design makes base gates predictable and easy to analyze. Each gate operation corresponds exactly to a mathematical unitary operation without additional complexity from modifiers.

### 4.2 Examples

**mqtref (Reference Semantics):**

```mlir
mqtref.x %q0                                    // Pauli-X gate
mqtref.rx %q0 {angle = 1.57 : f64}             // X-rotation with static parameter
mqtref.cx %q0, %q1                           // Controlled-NOT (2-qubit gate)
mqtref.u3 %q0 {theta = 0.0, phi = 0.0, lambda = 3.14159} // Generic single-qubit gate
```

**mqtopt (Value Semantics):**

```mlir
%q0_out = mqtopt.x %q0_in : !mqtopt.qubit
%q0_out = mqtopt.rx %q0_in {angle = 1.57 : f64} : !mqtopt.qubit
%q0_out, %q1_out = mqtopt.cx %q0_in, %q1_in : !mqtopt.qubit, !mqtopt.qubit
```

**Key Differences**: In mqtref, operations have side effects on qubit references. In mqtopt, operations consume input qubits and produce new output qubits, following SSA principles.

### 4.3 Parameterization Strategy

Gates support flexible parameterization to handle both compile-time and runtime parameters:

```mlir
// Static parameters (preferred for optimization)
mqtref.rx %q0 {angle = 1.57 : f64}

// Dynamic parameters (runtime values)
%angle = arith.constant 1.57 : f64
mqtref.rx %q0, %angle : f64

// Mixed parameters with mask indicating which are static
mqtref.u3 %q0, %runtime_theta {phi = 0.0, lambda = 3.14159, static_mask = [false, true, true]}
```

**Rationale**: Static parameters enable constant folding and symbolic computation at compile time, while dynamic parameters provide runtime flexibility. The mixed approach allows gradual specialization as more information becomes available during compilation.

## 5. Modifier Operations

Modifiers are wrapper operations that contain single-block regions holding the operations they modify. This design provides clean composition and nesting capabilities.

### 5.1 Control Operations (`ctrl` / `negctrl`)

Control operations add quantum control conditions to arbitrary quantum operations:

**mqtref (Reference Semantics):**

```mlir
// Single control
mqtref.ctrl %c0 {
  mqtref.x %q0  // Controlled-X (CNOT)
}

// Multiple controls
mqtref.ctrl %c0, %c1 {
  mqtref.x %q0  // Toffoli gate (CCX)
}

// Negative controls
mqtref.negctrl %c0 {
  mqtref.x %q0  // X gate triggered when c0 is |0⟩
}
```

**mqtopt (Value Semantics):**

```mlir
%c0_out, %q0_out = mqtopt.ctrl %c0_in {
  %q0_new = mqtopt.x %q0_in : !mqtopt.qubit
  mqtopt.yield %q0_new : !mqtopt.qubit
} : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)
```

**Rationale**: By treating controls as explicit wrapper operations, we can uniformly add controls to any quantum operation, including sequences and other modifiers. The region-based design makes the controlled operation explicit in the IR and enables easy analysis of the controlled subcomputation.

**Design Note**: In value semantics, control qubits must be threaded through the operation even though they're not modified, maintaining SSA form and enabling dataflow analysis.

### 5.2 Inverse Operations (`inv`)

Inverse operations compute the adjoint (Hermitian conjugate) of enclosed operations:

```mlir
// mqtref: Inverse of S gate (equivalent to S-dagger)
mqtref.inv {
  mqtref.s %q0
}

// mqtopt: Value-threaded inverse
%q0_out = mqtopt.inv {
  %q0_temp = mqtopt.s %q0_in : !mqtopt.qubit
  mqtopt.yield %q0_temp : !mqtopt.qubit
} : (!mqtopt.qubit) -> !mqtopt.qubit
```

**Rationale**: Making inverse explicit allows analysis passes to reason about adjoint relationships and enables optimizations like `inv(inv(X)) → X`. It also provides a uniform way to express inverse operations without requiring dedicated inverse variants of every gate.

### 5.3 Power Operations (`pow`)

Power operations compute fractional or integer powers of quantum operations:

```mlir
// Square root of X gate
mqtref.pow {exponent = 0.5 : f64} {
  mqtref.x %q0
}

// Dynamic exponent
%exp = arith.constant 0.25 : f64
mqtref.pow %exp {
  mqtref.ry %q0 {angle = 3.14159}
}
```

**Rationale**: Power operations are essential for quantum algorithms (e.g., quantum phase estimation, Grover's algorithm) and appear frequently in decompositions. Making them first-class enables direct representation without approximation.

### 5.4 Modifier Composition and Canonicalization

**Canonical Ordering**: To ensure consistent IR, we enforce a canonical nesting order:
`ctrl` (outermost) → `pow` → `inv` (innermost)

**Canonicalization Rules**:

- `inv(inv(X)) → X` (double inverse elimination)
- `pow(X, 1) → X` (identity power elimination)
- `pow(X, 0) → identity` (zero power simplification)
- `inv(pow(X, k)) → pow(inv(X), k)` (inverse-power commutation)
- `ctrl(inv(X)) → inv(ctrl(X))` when mathematically equivalent

**Rationale**: Canonical ordering prevents equivalent operations from having different representations, which would complicate analysis and optimization. The canonicalization rules capture mathematical identities that enable simplification.

## 6. Sequence Operations

Sequences group multiple quantum operations into logical units that can be analyzed and transformed as a whole.

### 6.1 Basic Sequences (Reference Semantics)

```mlir
mqtref.seq {
  mqtref.h %q0              // Hadamard gate
  mqtref.ctrl %q0 {         // Controlled operation
    mqtref.x %q1
  }
  mqtref.h %q0              // Another Hadamard
}
```

**Rationale**: Sequences provide a natural grouping mechanism for subcircuits that should be treated as units during optimization. They also enable hierarchical analysis - passes can choose to operate at the sequence level or dive into individual operations.

### 6.2 Value Semantics Threading

In `mqtopt`, sequences must explicitly thread all qubit values through the computation:

```mlir
%q0_out, %q1_out = mqtopt.seq %q0_in, %q1_in : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit) {
^entry(%q0: !mqtopt.qubit, %q1: !mqtopt.qubit):
  %q0_h = mqtopt.h %q0 : !mqtopt.qubit
  %q0_cx, %q1_cx = mqtopt.cx %q0_h, %q1 : !mqtopt.qubit, !mqtopt.qubit
  %q0_final = mqtopt.h %q0_cx : !mqtopt.qubit
  mqtopt.yield %q0_final, %q1_cx : !mqtopt.qubit, !mqtopt.qubit
}
```

**Explanation of ^entry syntax**: This is standard MLIR region syntax where `^entry` names the basic block and `(%q0: !mqtopt.qubit, %q1: !mqtopt.qubit)` declares the block arguments with their types. The sequence operation's operands become the block arguments, enabling proper SSA value threading within the region.

**Rationale**: Explicit value threading in sequences maintains SSA form and enables dataflow analysis. The block argument syntax is standard MLIR and clearly shows how values flow into and out of the sequence region.

## 7. Unified Interface Design

### 7.1 UnitaryOpInterface

All quantum operations implement a unified interface that abstracts away dialect differences:

```cpp
class UnitaryOpInterface {
public:
  // Gate identification
  StringRef getIdentifier();

  // Qubit counts
  size_t getNumTargets();
  size_t getNumQubits();

  // Control queries
  bool isControlled();
  size_t getNumPosControls();
  size_t getNumNegControls();
  size_t getNumControls();

  // Operand access
  OperandRange getTargetOperands();
  OperandRange getPosControlOperands();
  OperandRange getNegControlOperands();
  OperandRange getControlOperands();
  OperandRange getQubitOperands();
  OperandRange getOperands();

  // Result access (value semantics only)
  bool hasResults();
  ResultRange getTargetResults();
  ResultRange getPosControlResults();
  ResultRange getNegControlResults();
  ResultRange getControlResults();
  ResultRange getQubitResults();
  ResultRange getResults();

  // Mathematical representation
  UnitaryMatrix getUnitaryMatrix();
  bool hasStaticUnitary();

  // Parameter access
  size_t getNumParams();
  bool isParameterized();
  ArrayAttr getStaticParameters();
  OperandRange getDynamicParameters();
};
```

**Rationale**: This interface allows analysis passes to work uniformly across dialects without knowing whether they're dealing with reference or value semantics. The `hasResults()` method provides a clean way to branch when needed.

### 7.2 Implementation Strategy

- **Base Gates**: Implement interface directly with simple delegation to operands/results
- **Modifiers**: Delegate to wrapped operations with appropriate adjustments (e.g., ctrl adds to control count)
- **Sequences**: Provide aggregate information (total qubits touched, combined unitary matrix)

**Rationale**: This delegation pattern means that complex nested structures (e.g., controlled inverse sequences) automatically provide correct interface responses without manual bookkeeping.

## 8. User-Defined Gates

User-defined gates enable custom quantum operations with reusable definitions.

### 8.1 Matrix-Based Definitions

```mlir
// Define a custom single-qubit gate via its unitary matrix
mqtref.gate_def @pauli_y : tensor<2x2xcomplex<f64>> =
  dense<[[0.0+0.0i, 0.0-1.0i], [0.0+1.0i, 0.0+0.0i]]> : tensor<2x2xcomplex<f64>>

// Use the defined gate
mqtref.apply_gate @pauli_y %q0
```

**Rationale**: Matrix definitions provide an exact specification for custom gates, enabling precise simulation and analysis. They're particularly useful for gates that don't have natural decompositions into standard gates.

### 8.2 Composite Definitions

```mlir
// Define a gate as a sequence of existing operations
mqtref.gate_def @bell_prep %q0 : !mqtref.qubit, %q1 : !mqtref.qubit {
  mqtref.h %q0
  mqtref.cx %q0, %q1
}

// Apply the composite gate
mqtref.apply_gate @bell_prep %q0, %q1
```

**Rationale**: Composite definitions enable hierarchical design and code reuse. They can be inlined during lowering or kept as high-level constructs for analysis, depending on optimization needs.

### 8.3 Parameterized Gates

```mlir
// Define a parameterized rotation gate
mqtref.gate_def @custom_rotation %q : !mqtref.qubit attributes {params = ["theta", "phi"]} {
  mqtref.rz %q {angle = %phi}
  mqtref.ry %q {angle = %theta}
  mqtref.rz %q {angle = %phi}
}

// Apply with specific parameters
mqtref.apply_gate @custom_rotation %q0 {theta = 1.57 : f64, phi = 0.78 : f64}
```

**Rationale**: Parameterized gates enable template-like definitions that can be instantiated with different parameters, supporting gate libraries and algorithmic patterns.

## 9. Parser Sugar and Builder APIs

### 9.1 Parser Sugar for Common Patterns

Instead of complex chaining syntax, we provide natural abbreviations for frequent patterns:

**Controlled Gate Shortcuts:**

```mlir
// Standard controlled gates (parser expands these automatically)
mqtref.cx %c, %t        // → mqtref.ctrl %c { mqtref.x %t }
mqtref.cz %c, %t        // → mqtref.ctrl %c { mqtref.z %t }
mqtref.ccx %c0, %c1, %t // → mqtref.ctrl %c0, %c1 { mqtref.x %t }

// Multi-controlled variants
mqtref.mcx (%c0, %c1, %c2), %t  // → mqtref.ctrl %c0, %c1, %c2 { mqtref.x %t }
mqtref.mcp (%c0, %c1), %t {phase = 1.57}  // Controlled phase with multiple controls
```

**Rationale**: This approach provides ergonomic shortcuts for common cases without introducing complex chaining operators. The shortcuts expand to the full form during parsing, so all downstream processing sees the canonical representation.

### 9.2 C++ Builder API

```cpp
// Fluent builder interface
class QuantumCircuitBuilder {
public:
  // Basic gates with natural names
  QuantumCircuitBuilder& x(Value qubit);
  QuantumCircuitBuilder& h(Value qubit);
  QuantumCircuitBuilder& cx(Value control, Value target);

  // Modifier combinators
  QuantumCircuitBuilder& ctrl(ValueRange controls, std::function<void()> body);
  QuantumCircuitBuilder& inv(std::function<void()> body);
  QuantumCircuitBuilder& pow(double exponent, std::function<void()> body);

  // Convenient combinations
  QuantumCircuitBuilder& ccx(Value c1, Value c2, Value target);
  QuantumCircuitBuilder& toffoli(Value c1, Value c2, Value target) { return ccx(c1, c2, target); }
};

// Example usage
QuantumCircuitBuilder builder(mlirBuilder, location);
builder.h(q0)
       .ctrl({c0}, [&]() { builder.x(q1); })
       .ccx(c0, c1, q2);
```

**Rationale**: The builder API provides a natural C++ interface that maps cleanly to the IR structure. Lambda functions for modifier bodies give clear scoping and enable complex nested structures.

## 10. Analysis and Optimization Infrastructure

### 10.1 UnitaryMatrix Support Library

```cpp
class UnitaryMatrix {
private:
  // Efficient representations for common cases
  std::variant<
    Matrix2x2,      // Single-qubit gates
    Matrix4x4,      // Two-qubit gates
    SymbolicExpr,   // Larger or parameterized gates
    LazyProduct     // Composition chains
  > representation;

public:
  // Composition operations
  UnitaryMatrix compose(const UnitaryMatrix& other) const;
  UnitaryMatrix adjoint() const;
  UnitaryMatrix power(double exponent) const;
  UnitaryMatrix control(unsigned num_controls) const;

  // Materialization
  DenseElementsAttr toDenseElements(MLIRContext* ctx) const;
  bool isIdentity() const;
  bool isUnitary() const;  // Verification
};
```

**Rationale**: This library provides efficient computation for the unitary matrices that drive quantum optimization. The multi-representation approach keeps small matrices fast while supporting larger compositions through symbolic expressions.

### 10.2 Canonicalization Passes

**NormalizationPass**: Enforces canonical modifier ordering and eliminates redundancies

- Reorders nested modifiers to canonical form (ctrl → pow → inv)
- Eliminates identity operations (`pow(X, 1) → X`, `inv(inv(X)) → X`)
- Merges adjacent compatible modifiers
- Simplifies power modifiers wherever feasible (`pow(RZ(pi/2), 2) → RZ(pi)`)

**SequenceFlatteningPass**: Inlines nested sequences when beneficial

- Removes sequence boundaries that don't provide optimization barriers
- Preserves sequences that are referenced by symbols or have special annotations

**ConstantFoldingPass**: Folds static parameters and eliminates trivial operations

- Combines adjacent rotations with static angles
- Eliminates gates with zero rotation angles
- Evaluates static matrix expressions

**Rationale**: These passes work together to keep the IR in a canonical form that simplifies subsequent analysis and optimization. Each pass has a focused responsibility and can be run independently or as part of a pipeline.

## 11. Dialect Conversions

### 11.1 mqtref ↔ mqtopt Conversion

The conversion between reference and value semantics requires careful handling of SSA values and side effects.

**mqtref → mqtopt (Adding SSA Values)**:

```cpp
// Pattern: Convert side-effect operations to value-producing operations
mqtref.x %q → %q_new = mqtopt.x %q : !mqtopt.qubit

// Control conversion requires threading control qubits
mqtref.ctrl %c { mqtref.x %t } →
  %c_out, %t_out = mqtopt.ctrl %c_in {
    %t_new = mqtopt.x %t_in : !mqtopt.qubit
    mqtopt.yield %t_new : !mqtopt.qubit
  } : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)
```

**mqtopt → mqtref (Removing SSA Values)**:

```cpp
// Drop results and convert to side-effect form
%q_out = mqtopt.x %q_in : !mqtopt.qubit → mqtref.x %q_in

// Ensure proper dominance and liveness in the converted code
```

**Key Challenges**:

- **SSA Value Threading**: mqtopt requires explicit threading of all qubit values

**Rationale**: These conversions enable using the same quantum algorithm in different contexts - mqtref for simulation and debugging, mqtopt for optimization and compilation to hardware.

### 11.2 Lowering to Hardware Targets

Standard conversion patterns lower high-level operations to target-specific instruction sets:

```cpp
// Example: Lowering controlled gates to native basis
mqtref.ctrl %c { mqtref.ry %t {angle = θ} } →
  mqtref.rz %t {angle = θ/2}
  mqtref.cx %c, %t
  mqtref.rz %t {angle = -θ/2}
  mqtref.cx %c, %t
```

**Rationale**: Hardware targets have limited native gate sets, so high-level operations must be decomposed into available primitives while preserving mathematical equivalence.

## 12. Testing Strategy

Generally, it should be part of this endeavor to come up with a testing strategy that we can exercise across our efforts going forward.
It has already become quite clear that we do not want to extensively write FileCheck strings as it is very error prone. We are currently likely spending more time on fixing FileCheck strings than actually developing features.
Hence, even the integration tests down below should be considered to be realized differently in C++.

### 12.1 Unit Tests (C++)

- **Interface implementations**: Verify UnitaryOpInterface methods return consistent results
- **UnitaryMatrix library operations**: Test composition, adjoint, and power operations
- **Builder API functionality**: Ensure generated IR matches expected patterns

### 12.2 Integration Tests (LIT)

- **Parser/printer round-trips**: Verify text representation preserves semantics
- **Canonicalization correctness**: Test that canonical forms are stable and unique
- **Conversion patterns**: Verify dialect conversions preserve quantum semantics
- **Error handling and verification**: Test malformed IR is properly rejected

### 12.3 Quantum Semantics Tests

```mlir
// RUN: mlir-opt %s -test-quantum-canonicalize | FileCheck %s

// Test double inverse elimination
func.func @test_double_inverse() {
  %q = mqtref.alloc : !mqtref.qubit

  // CHECK: mqtref.x %{{.*}}
  // CHECK-NOT: mqtref.inv
  mqtref.inv {
    mqtref.inv {
      mqtref.x %q
    }
  }
  return
}

// Test control threading in value semantics
func.func @test_control_threading() {
  %c = mqtopt.alloc : !mqtopt.qubit
  %t = mqtopt.alloc : !mqtopt.qubit

  // CHECK: %[[C_OUT:.*]], %[[T_OUT:.*]] = mqtopt.ctrl %{{.*}} {
  // CHECK:   %[[T_NEW:.*]] = mqtopt.x %{{.*}}
  // CHECK:   mqtopt.yield %[[T_NEW]]
  // CHECK: }
  %c_out, %t_out = mqtopt.ctrl %c {
    %t_new = mqtopt.x %t : !mqtopt.qubit
    mqtopt.yield %t_new : !mqtopt.qubit
  } : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)

  return
}
```

**Rationale**: Quantum semantics are subtle and error-prone. Comprehensive testing with both positive and negative test cases ensures the implementation correctly handles edge cases and maintains mathematical correctness.

## 13. Conclusion

This comprehensive redesign addresses the fundamental limitations of the current quantum MLIR infrastructure while positioning it for future growth. The key innovations—compositional modifiers, unified interfaces, and enhanced ergonomics—will significantly improve developer productivity and enable more sophisticated quantum compiler optimizations.

The modular implementation plan reduces project risk by maintaining working functionality at each milestone. The emphasis on testing and performance ensures that the new system will be both reliable and efficient.

By aligning with MLIR best practices and providing clean abstractions, this design creates a solid foundation for quantum compiler research and development within the MQT ecosystem.

The expected outcome is a quantum compiler infrastructure that is easier to use, more analyzable, and better positioned for the evolving needs of quantum computing research and application development.
