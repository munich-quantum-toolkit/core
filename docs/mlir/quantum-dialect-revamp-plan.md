## 1. Overview and Goals

### 1.1 Current State

We have two quantum dialects:

- `mqtref`: Reference semantics (side-effect based)
- `mqtopt`: Value semantics (SSA based)

Both currently provide basic gates and quantum operations but suffer from:

- Verbose / inconsistent builder ergonomics (especially for controlled forms)
- Modifier semantics encoded inconsistently (implicit attributes, ad‑hoc controls)
- Fragmented interfaces for querying structural properties
- Limited composability for combined modifiers (e.g., controlled powers of inverses)

### 1.2 Proposed Direction

This revamp establishes a uniform, compositional IR that:

1. Treats all gate modifiers (controls, negative controls, powers, inverses) as explicit wrapper ops with regions.
2. Provides a single structural interface (`UnitaryOpInterface`) across both dialects.
3. Normalizes modifier ordering, merging, and identity elimination early and deterministically.
4. Supports both a rich named gate set AND matrix-based unitaries.
5. Emphasizes builder-based structural/semantic testing over brittle text-based checks.

---

## 2. Architecture Overview

### 2.1 Dialect Structure

```
Common/
├── Interfaces (UnitaryOpInterface, etc.)
├── Traits (Hermitian, Diagonal, SingleTarget, etc.)
└── Support (UnitaryExpr / UnitaryMatrix utilities)

mqtref/
├── Types (qubit)
├── Base Gates (x, rx, cx, u1/u2/u3, etc.)
├── Modifiers (ctrl, negctrl, pow, inv)
├── Sequences (seq)
└── Resources (alloc, measure, ...)

mqtopt/
└── Parallel set with value semantics (results thread through)
```

### 2.2 Principles

1. Base gate ops contain only their target operands and parameters (no embedded controls / modifiers).
2. Modifiers are explicit region wrapper ops enabling arbitrary nesting in canonical order.
3. Interface / traits provide uniform structural queries (control counts, target counts, matrix form when available).
4. Reference vs value semantics differences are localized to operation signatures & result threading.

**Rationale:** Predictable structural form enables simpler canonicalization, hashing, CSE, and semantic equivalence checks.

---

## 3. Types and Memory Model

### 3.1 Quantum Types

- `!mqtref.qubit`: Stateful reference (in-place mutation semantics).
- `!mqtopt.qubit`: SSA value (must be threaded; operations consume & produce qubits).

### 3.2 Register Handling

Standard MLIR types (e.g., `memref`) manage collections:

```mlir
%qreg = memref.alloc() : memref<2x!mqtref.qubit>
%q0 = memref.load %qreg[%c0] : memref<2x!mqtref.qubit>
%creg = memref.alloc() : memref<2xi1>
%bit = mqtref.measure %q0 : i1
memref.store %bit, %creg[%c0] : memref<2xi1>
```

---

## 4. Base Gate Operations

### 4.1 Philosophy

Minimal, parameterized primitives; all composition via wrappers.

### 4.2 Examples (Reference Semantics)

```mlir
mqtref.x %q0
mqtref.rx %q0 {angle = 1.57 : f64}
mqtref.cx %q0, %q1
mqtref.u3 %q0 {theta = 0.0 : f64, phi = 0.0 : f64, lambda = 3.14159 : f64}
```

### 4.3 Examples (Value Semantics)

```mlir
%q0_out = mqtopt.x %q0_in : !mqtopt.qubit
%q0_out = mqtopt.rx %q0_in {angle = 1.57 : f64} : !mqtopt.qubit
%q0_out, %q1_out = mqtopt.cx %q0_in, %q1_in : !mqtopt.qubit, !mqtopt.qubit
```

### 4.4 Parameterization Strategy

```mlir
mqtref.rx %q0 {angle = 1.57 : f64}            // static
%theta = arith.constant 1.57 : f64
mqtref.rx %q0, %theta : f64                   // dynamic
%dyn_theta = arith.constant 0.5 : f64
mqtref.u3 %q0, %dyn_theta {phi = 0.0 : f64, lambda = 3.14159 : f64,
                           static_mask = [false, true, true]}
```

---

## 5. Modifier Operations

Modifiers are single-region wrappers; canonical nesting: `ctrl` → `negctrl` → `pow` → `inv`.

### 5.1 Controls (`ctrl`) & Negative Controls (`negctrl`)

Reference semantics:

```mlir
mqtref.ctrl %c0 {
  mqtref.x %t0           // CNOT
}

mqtref.ctrl %c0, %c1 {
  mqtref.x %t0           // Toffoli (CCX)
}

mqtref.negctrl %c0 {
  mqtref.x %t0           // Fires when %c0 is |0>
}
```

Value semantics (control + target operands; region blocks thread values explicitly):

```mlir
%c_out, %t_out = mqtopt.ctrl %c_in, %t_in : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit) {
^entry(%c: !mqtopt.qubit, %t: !mqtopt.qubit):
  %t_new = mqtopt.x %t : !mqtopt.qubit
  mqtopt.yield %c, %t_new : !mqtopt.qubit, !mqtopt.qubit
}
```

Negative controls are NOT rewritten into positive controls (no implicit X-sandwich canonicalization).

### 5.2 Power (`pow`)

```mlir
mqtref.pow {exponent = 0.5 : f64} {
  mqtref.x %q0   // sqrt(X)
}

%exp = arith.constant 0.25 : f64
mqtref.pow %exp {
  mqtref.ry %q0 {angle = 3.14159 : f64}
}
```

Value semantics single-qubit power:

```mlir
%q_out = mqtopt.pow {exponent = 0.5 : f64} %q_in : (!mqtopt.qubit) -> !mqtopt.qubit {
^entry(%q: !mqtopt.qubit):
  %q_x = mqtopt.x %q : !mqtopt.qubit
  mqtopt.yield %q_x : !mqtopt.qubit
}
```

### 5.3 Inverse (`inv`)

```mlir
mqtref.inv {
  mqtref.s %q0    // S†
}

%q0_out = mqtopt.inv %q0_in : (!mqtopt.qubit) -> !mqtopt.qubit {
^entry(%q0: !mqtopt.qubit):
  %q0_s = mqtopt.s %q0 : !mqtopt.qubit
  mqtopt.yield %q0_s : !mqtopt.qubit
}
```

### 5.4 Nested Modifier Example (Canonical Order)

Reference semantics nested chain:

```mlir
mqtref.ctrl %c0, %c1 {
  mqtref.negctrl %c2 {
    mqtref.pow {exponent = 2.0 : f64} {
      mqtref.inv {
        mqtref.x %t0
      }
    }
  }
}
```

Value semantics counterpart (illustrative; block arguments explicit at each nesting level):

```mlir
%c_out, %t_out = mqtopt.ctrl %c_in, %t_in : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit) {
^entry(%c: !mqtopt.qubit, %t: !mqtopt.qubit):
  %c_neg_out, %t_neg = mqtopt.negctrl %c, %t : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit) {
  ^entry(%cn: !mqtopt.qubit, %tn: !mqtopt.qubit):
    %t_pow = mqtopt.pow {exponent = 2.0 : f64} %tn : (!mqtopt.qubit) -> !mqtopt.qubit {
      ^entry(%tp: !mqtopt.qubit):
        %t_inv = mqtopt.inv %tp : (!mqtopt.qubit) -> !mqtopt.qubit {
          ^entry(%ti: !mqtopt.qubit):
            %t_x = mqtopt.x %ti : !mqtopt.qubit
            mqtopt.yield %t_x : !mqtopt.qubit
        }
        mqtopt.yield %t_inv : !mqtopt.qubit
    }
    mqtopt.yield %cn, %t_pow : !mqtopt.qubit, !mqtopt.qubit
  }
  mqtopt.yield %c_neg_out, %t_neg : !mqtopt.qubit, !mqtopt.qubit
}
```

(Assumption: Each wrapper result list mirrors its operand list order for pass-through + transformed targets.)

### 5.5 Modifier Semantics Summary

- Control count = sum of outer wrappers + inner (flattened by normalization).
- `pow` & `inv` forward control counts unchanged.
- No negctrl → ctrl auto-normalization.

---

## 6. Sequence Operations (`seq`)

Sequences group subcircuits; they are control-neutral (always 0 controls) by decision.

Reference semantics:

```mlir
mqtref.seq {
  mqtref.h %q0
  mqtref.ctrl %q0 {
    mqtref.x %q1
  }
  mqtref.h %q0
}
```

Value semantics:

```mlir
%q0_out, %q1_out = mqtopt.seq %q0_in, %q1_in : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit) {
^entry(%q0: !mqtopt.qubit, %q1: !mqtopt.qubit):
  %q0_h = mqtopt.h %q0 : !mqtopt.qubit
  %q0_cx, %q1_cx = mqtopt.cx %q0_h, %q1 : !mqtopt.qubit, !mqtopt.qubit
  %q0_final = mqtopt.h %q0_cx : !mqtopt.qubit
  mqtopt.yield %q0_final, %q1_cx : !mqtopt.qubit, !mqtopt.qubit
}
```

---

## 7. Unified Interface Design

### 7.1 `UnitaryOpInterface` (Conceptual Extract)

```cpp
class UnitaryOpInterface {
public:
  StringRef getIdentifier();
  size_t getNumTargets();
  size_t getNumQubits();
  size_t getNumPosControls();
  size_t getNumNegControls();
  size_t getNumControls();           // pos + neg (seq = 0)
  OperandRange getTargetOperands();
  OperandRange getPosControlOperands();
  OperandRange getNegControlOperands();
  OperandRange getControlOperands(); // concatenated
  OperandRange getQubitOperands();
  bool hasResults();
  ResultRange getTargetResults();
  // Control results present in value semantics wrappers
  UnitaryMatrix getUnitaryMatrix();  // Static or computed
  bool hasStaticUnitary();
  size_t getNumParams();
  ArrayAttr getStaticParameters();
  OperandRange getDynamicParameters();
};
```

### 7.2 Notes

- No explicit enumeration for pow/inv stack layers (decision: unnecessary for MVP).
- Sequences report 0 controls even if containing controlled ops.
- Matrix or composite definitions report 0 intrinsic controls; wrappers add controls.

---

## 8. Matrix & User-Defined Gates

Matrix-based unitary expression is part of MVP (NOT deferred).

### 8.1 Matrix Unitary Operation

Attribute: flat row‑major dense tensor `tensor<(2^(n)*2^(n)) x complex<f64>>`.

2×2 example (Pauli-Y):

```mlir
// Row-major: [0  -i ; i  0]
mqtref.unitary %q0 { matrix = dense<[0.0+0.0i, 0.0-1.0i, 0.0+1.0i, 0.0+0.0i]> : tensor<4xcomplex<f64>> }
```

Value semantics:

```mlir
%q0_out = mqtopt.unitary %q0_in { matrix = dense<[0.0+0.0i, 0.0-1.0i, 0.0+1.0i, 0.0+0.0i]> : tensor<4xcomplex<f64>> } : !mqtopt.qubit
```

4×4 example (identity on two qubits):

```mlir
mqtref.unitary %q0, %q1 { matrix = dense<[
  1.0+0.0i, 0.0+0.0i, 0.0+0.0i, 0.0+0.0i,
  0.0+0.0i, 1.0+0.0i, 0.0+0.0i, 0.0+0.0i,
  0.0+0.0i, 0.0+0.0i, 1.0+0.0i, 0.0+0.0i,
  0.0+0.0i, 0.0+0.0i, 0.0+0.0i, 1.0+0.0i]> : tensor<16xcomplex<f64>> }
```

### 8.2 Gate Definitions (Symbolic / Composite)

```mlir
// Composite definition
mqtref.gate_def @bell_prep %a : !mqtref.qubit, %b : !mqtref.qubit {
  mqtref.h %a
  mqtref.cx %a, %b
}

// Application
mqtref.apply_gate @bell_prep %q0, %q1
```

### 8.3 Parameterized Definitions

```mlir
// Parameters modeled as additional operands (value semantics for numeric params)
mqtref.gate_def @custom_rotation(%q: !mqtref.qubit, %theta: f64, %phi: f64) {
  mqtref.rz %q, %phi : f64
  mqtref.ry %q, %theta : f64
  mqtref.rz %q, %phi : f64
}

%theta = arith.constant 1.57 : f64
%phi    = arith.constant 0.78 : f64
mqtref.apply_gate @custom_rotation %q0, %theta, %phi : (!mqtref.qubit, f64, f64)
```

Definitions themselves are control-neutral; wrapping ops add controls.

### 8.4 Unitarity Validation

A dedicated validation pass (Matrix Unitary Validation) verifies numerical unitarity within tolerance; strict mode (optional) can enforce tighter bounds.

---

## 9. Parser Sugar & Builder APIs

### 9.1 Parser Sugar Examples

```mlir
// Sugar → canonical expansion
mqtref.cx %c, %t          // expands to: mqtref.ctrl %c { mqtref.x %t }
mqtref.cz %c, %t          // expands to: mqtref.ctrl %c { mqtref.z %t }
mqtref.ccx %c0, %c1, %t   // expands to: mqtref.ctrl %c0, %c1 { mqtref.x %t }
```

### 9.2 C++ Builder (Sketch)

```cpp
class QuantumCircuitBuilder {
public:
  QuantumCircuitBuilder &x(Value q);
  QuantumCircuitBuilder &h(Value q);
  QuantumCircuitBuilder &cx(Value c, Value t);
  QuantumCircuitBuilder &ctrl(ValueRange controls, function_ref<void()> body);
  QuantumCircuitBuilder &negctrl(ValueRange controls, function_ref<void()> body);
  QuantumCircuitBuilder &pow(double exponent, function_ref<void()> body);
  QuantumCircuitBuilder &inv(function_ref<void()> body);
};

builder.h(q0)
       .ctrl({c0}, [&](){ builder.x(q1); })
       .ctrl({c0, c1}, [&](){ builder.x(q2); });
```

---

## 10. Canonicalization & Transformation Stages

### 10.1 Normalization (Early Canonical Form)

Responsibilities (always first):

- Enforce modifier order `ctrl` → `negctrl` → `pow` → `inv`.
- Flatten nested same-kind control wrappers (aggregate control lists) while preserving neg/pos distinction.
- Merge adjacent compatible `pow` / eliminate identities (`pow(exp=1)`, `inv(inv(X))`).
- Algebraic simplifications (e.g., `pow(RZ(pi/2), 2) → RZ(pi)`).
- Remove dead identity operations (e.g., `pow(exp=0)` → (implicit identity) if allowed by semantics).
- Prepare IR for subsequent named gate passes.

### 10.2 Named Simplification

- Convert `U3/U2/U1` parameter sets to simplest named gate when within numeric tolerance.
- Improves readability & hashing stability.

### 10.3 Universal Expansion

- Expand named single-qubit gates to canonical `U3` (backend / pipeline selectable).
- Typically applied only in universal backends or downstream lowering flows.

### 10.4 Matrix Validation

- Verify matrix attribute size matches `2^(n)*2^(n)`.
- Check numerical unitarity within tolerance (fast path for 2×2 & 4×4).

### 10.5 (Deferred) Matrix Decomposition

- Future: Decompose large / arbitrary matrix unitaries into basis gates (NOT in MVP).

**No pass rewrites `negctrl` into positive controls.**

---

## 11. Pass Inventory

| Pass                        | Purpose                                                                              | Phase                    | Idempotent               | Mandatory                   | Notes                                 |
| --------------------------- | ------------------------------------------------------------------------------------ | ------------------------ | ------------------------ | --------------------------- | ------------------------------------- |
| NormalizationPass           | Enforce modifier order; merge & simplify; flatten controls; early algebraic cleanups | Canonicalization (first) | Yes                      | Yes                         | Integrated with canonicalize pipeline |
| NamedSimplificationPass     | Replace `U3/U2/U1` with simplest named gate                                          | Simplification           | Yes (post-normalization) | Baseline (recommended)      | Tolerance-based                       |
| UniversalExpansionPass      | Expand named gate → `U3`                                                             | Lowering / Backend Prep  | Yes                      | Optional                    | Backend / pipeline controlled         |
| MatrixUnitaryValidationPass | Verify matrix size + unitarity                                                       | Verification / Early     | Yes                      | Yes (if matrix ops present) | Fast paths 2×2 & 4×4                  |

### Deferred (Not MVP)

- Matrix decomposition pass
- Basis gate registry
- Extended trait inference for composites & matrix ops
- Advanced symbolic parameter algebra

---

## 12. Dialect Conversions

### 12.1 `mqtref` → `mqtopt`

```cpp
// Side-effect → value threading
mqtref.x %q  →  %q_new = mqtopt.x %q : !mqtopt.qubit

// Controlled example
mqtref.ctrl %c { mqtref.x %t } →
  %c_out, %t_out = mqtopt.ctrl %c, %t {
    %t_new = mqtopt.x %t : !mqtopt.qubit
    mqtopt.yield %t_new : !mqtopt.qubit
  } : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)
```

### 12.2 `mqtopt` → `mqtref`

```cpp
%q_out = mqtopt.x %q_in : !mqtopt.qubit  →  mqtref.x %q_in
// Drop result, preserve ordering & effects
```

**Challenges:** Ensuring correct dominance & preserving semantic ordering during un-nesting / region translation.

---

## 13. Testing Strategy

Priority shift: structural & semantic equivalence via builders (googletest) > textual pattern checks.

### 13.1 Structural / Semantic Tests (Primary)

- Use IR builders to construct original & expected forms, run normalization + optional passes, then compare via:
  - Shape / op sequence equivalence (ignoring SSA names).
  - Control & target counts via `UnitaryOpInterface`.
  - Optional unitary matrix equivalence (numerical tolerance) for small ops.
- Idempotence: Run NormalizationPass twice; assert no further changes.

### 13.2 Parser / Printer Smoke (Minimal Textual Tests)

- Round-trip for: base gates, each modifier, nested modifier chain, matrix unitary, composite definition, sequence.
- FileCheck limited to presence/absence of key ops (avoid brittle SSA checks).

Example smoke test snippet:

```mlir
// CHECK-LABEL: func @nested_mods
func.func @nested_mods(%c: !mqtref.qubit, %n: !mqtref.qubit, %t: !mqtref.qubit) {
  mqtref.ctrl %c {
    mqtref.negctrl %n {
      mqtref.pow {exponent = 2.0 : f64} {
        mqtref.inv { mqtref.x %t }
      }
    }
  }
  return
}
```

### 13.3 Utility Helpers (C++)

- `assertEquivalent(Op a, Op b, EquivalenceOptions opts)`
- `buildNestedModifiers(builder, patternSpec)`
- Matrix validation harness (inject near-unitary perturbations and assert failures).

### 13.4 Negative Tests

- Malformed matrix size
- Non-unitary matrix beyond tolerance
- Disallowed modifier order (e.g., `inv` wrapping `ctrl` directly → should be reordered by normalization)

### 13.5 Coverage Summary

All ops (base, modifiers, sequences, unitary/matrix, composite definitions) must have:

- Builder creation tests
- Normalization idempotence tests
- Interface query sanity tests

---

## 14. Analysis & Optimization Infrastructure

### 14.1 `UnitaryMatrix` Utility (Sketch)

```cpp
class UnitaryMatrix {
  // Variant representations: small fixed (2x2, 4x4), symbolic, lazy product
public:
  UnitaryMatrix compose(const UnitaryMatrix &rhs) const;
  UnitaryMatrix adjoint() const;
  UnitaryMatrix power(double exponent) const;
  UnitaryMatrix control(unsigned numPos, unsigned numNeg) const;
  DenseElementsAttr toDenseElements(MLIRContext* ctx) const;
  bool isIdentity() const;
  bool isUnitary(double tol = 1e-10) const;
};
```

Fast paths for 2×2 & 4×4 feed both transformation heuristics and validation.

---

## 15. Canonical Identities & Algebraic Rules (Non-Exhaustive)

Guaranteed (within normalization) where semantics preserved:

- `inv(inv(X)) → X`
- `pow(X, 1) → X`
- `pow(X, 0) → identity` (subject to representation policy; may drop operation)
- `pow(inv(X), k) ↔ inv(pow(X, k))` (normalized placement enforces `pow` outside `inv` → `pow(inv(X), k)` canonicalizes to `pow` wrapping `inv` only if order rule maintained)
- Consecutive `pow` merges: `pow(pow(X, a), b) → pow(X, a*b)`
- Control aggregation: nested `ctrl` of `ctrl` flattens; same for nested `negctrl`; mixed pos/neg preserve order lists separately.

No rule rewrites `negctrl` into a positive control sandwich.

---

## 16. Conclusion

The revamp solidifies a compositional, analyzable, and canonical quantum IR:

- Modifier wrappers with a fixed canonical order including explicit negative controls.
- Early, mandatory normalization ensuring deterministic structure for all downstream passes.
- Rich named gate ecosystem retained alongside matrix-based unitaries (with validation) in the MVP.
- Bidirectional single-qubit canonicalization allows readability and backend flexibility.
- Testing focus shifts to robust builder-based structural equivalence; textual tests minimized.

This plan supersedes earlier notions that matrix gates or negative controls might be deferred or normalized away. The resulting infrastructure provides a stable foundation for future extensions (matrix decomposition, basis registries, advanced trait inference) without compromising current clarity or performance.

No open questions remain for the MVP scope defined here.
