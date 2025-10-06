# Unitary Expression Support Library

**Purpose:** Provide efficient unitary matrix computation and symbolic reasoning for transformation and optimization passes in the MQT MLIR compilation infrastructure.

## Overview

The `UnitaryExpr` support library is a lightweight C++20 framework that enables transformation passes to efficiently reason about, compose, and manipulate quantum gate unitaries.
It serves as the computational backbone for operations like static matrix extraction, gate fusion, equivalence checking, and optimization within the MQT MLIR dialects.

This library integrates seamlessly with the unified `UnitaryOpInterface` defined in the quantum dialect architecture, providing the underlying computational machinery for extracting and manipulating unitary matrices from quantum operations.

## Design Philosophy

The library balances performance, expressiveness, and integration with MLIR infrastructure through the following principles:

### Core Principles

- **Performance-Critical:** Optimized for small matrices common in quantum computing (2×2 single-qubit, 4×4 two-qubit gates)
- **MLIR-Native:** Leverages MLIR/LLVM support types (`mlir::ArrayRef`, `mlir::SmallVector`, `llvm::APFloat`) over standard library equivalents where appropriate
- **Allocation-Aware:** Stack-based storage for fixed-size matrices; minimal heap allocations for dynamic cases
- **Inline-Friendly:** Header-only implementations for critical paths to enable aggressive compiler optimization
- **Type-Safe:** Exploits C++20 concepts and the type system to prevent dimension mismatches at compile time
- **Symbolic When Needed:** Supports both concrete materialized matrices and symbolic expression trees for deferred evaluation
- **Zero External Dependencies:** Relies exclusively on LLVM/MLIR infrastructure—no Eigen, BLAS, or quantum simulation libraries

### Design Goals

The library addresses three key use cases in quantum compilation:

1. **Static Analysis:** Enable passes to query matrix properties (hermiticity, diagonality, unitarity) without full materialization
2. **Gate Fusion:** Efficiently compose sequences of gates to identify fusion opportunities
3. **Equivalence Checking:** Compare unitary operations for semantic equivalence during optimization

## Qubit Ordering and Endianness

A fundamental challenge in multi-qubit gate composition is correctly handling qubit ordering and endianness conventions. The matrix representation of a quantum gate depends on which physical qubits it operates on and in what order they appear in the tensor product basis.

### The Problem

Consider a controlled-X gate applied to qubits `q0` and `q1`:

```mlir
quartz.ctrl(%q0) { quartz.x %q1 }  // q0 controls, q1 is target
```

versus

```mlir
quartz.ctrl(%q1) { quartz.x %q0 }  // q1 controls, q0 is target
```

Even though both operations have the same SSA values, their matrix representations differ because the computational basis ordering changes:

- First case: basis ordered as `|q0⟩ ⊗ |q1⟩` → states `|00⟩, |01⟩, |10⟩, |11⟩`
- Second case: basis ordered as `|q1⟩ ⊗ |q0⟩` → states `|00⟩, |01⟩, |10⟩, |11⟩` (but q1 is now the high-order bit)

This becomes especially critical when composing sequences of multi-qubit gates where qubits appear in different positions.

### Endianness Convention

The library adopts **big-endian** qubit ordering convention, which aligns with standard quantum computing literature and notation:

- **Qubit position 0** (first in the list) corresponds to the **most significant bit** in the basis state
- **Qubit position n-1** (last in the list) corresponds to the **least significant bit** in the basis state
- For n qubits at positions [q₀, q₁, ..., qₙ₋₁], the basis state index is: 2^(n-1)·q₀ + 2^(n-2)·q₁ + ... + 2^0·qₙ₋₁
- Basis states are enumerated in standard binary order: `|00⟩, |01⟩, |10⟩, |11⟩` for two qubits

**Example:** For `ctrl(%q0) { x %q1 }` where the qubit order is `[q0, q1]`:

- `q0` at position 0 (MSB) → controls on bit 1 (high-order bit)
- `q1` at position 1 (LSB) → target is bit 0 (low-order bit)
- Tensor product: `|q0⟩ ⊗ |q1⟩`
- Matrix acts as: `|00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩` (standard CNOT with q0 control, q1 target)

**Rationale:** Big-endian ordering is the standard convention in quantum computing papers and textbooks, where multi-qubit states are written left-to-right with the first qubit as the leftmost tensor factor: `|ψ⟩ = |q₀⟩ ⊗ |q₁⟩ ⊗ ... ⊗ |qₙ₋₁⟩`. This convention ensures that the library's matrix representations match published results and established quantum algorithms.

### Design Principles

1. **Explicit Qubit Ordering:** All multi-qubit operations must explicitly track which qubits they operate on and in what order
2. **Permutation Tracking:** When composing gates with different qubit orderings, explicit permutation matrices handle reordering
3. **Canonical Forms:** Operations on the same qubits in the same order can be directly composed; otherwise, reordering is required
4. **Efficient Common Case:** Same-order composition (the common case) has zero overhead; reordering only when necessary

### Qubit Order Representation

```cpp
namespace mqt::unitary {

/// Represents an ordered list of qubits for a multi-qubit operation
/// Uses MLIR Value to represent SSA values (for Flux) or qubit references (for Quartz)
using QubitOrder = mlir::SmallVector<mlir::Value, 4>;

/// Check if two qubit orders are identical (same qubits in same positions)
[[nodiscard]] inline bool isSameOrder(const QubitOrder& a, const QubitOrder& b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin());
}

/// Check if two qubit orders operate on the same qubits (possibly different order)
[[nodiscard]] bool isSameQubits(const QubitOrder& a, const QubitOrder& b);

/// Compute permutation mapping from order 'from' to order 'to'
/// Returns std::nullopt if they don't contain the same qubits
[[nodiscard]] std::optional<mlir::SmallVector<size_t, 4>>
  computePermutation(const QubitOrder& from, const QubitOrder& to);

} // namespace mqt::unitary
```

## Core Data Structures

### Fixed-Size Matrix Types

The library provides specialized matrix types for the most common quantum operations, leveraging stack allocation and cache-friendly memory layouts.

#### Mat2: Single-Qubit Matrices

The `Mat2` class represents 2×2 complex matrices for single-qubit gates:

```cpp
namespace mqt::unitary {

/// 2×2 complex matrix for single-qubit gates (stack-allocated, 64 bytes)
class Mat2 {
public:
  using Complex = std::complex<double>;

  /// Stack storage: 4 complex values = 8 doubles = 64 bytes
  /// Fits in a single CPU cache line
  std::array<Complex, 4> data;

  /// Constructors
  constexpr Mat2() = default;
  constexpr Mat2(Complex a00, Complex a01, Complex a10, Complex a11);

  /// Static factory methods for common gates
  static constexpr Mat2 identity() noexcept;
  static constexpr Mat2 pauliX() noexcept;
  static constexpr Mat2 pauliY() noexcept;
  static constexpr Mat2 pauliZ() noexcept;
  static constexpr Mat2 hadamard() noexcept;
  static constexpr Mat2 phaseS() noexcept;
  static constexpr Mat2 phaseT() noexcept;

  /// Element access (row-major ordering)
  constexpr Complex& operator()(size_t row, size_t col) noexcept;
  constexpr const Complex& operator()(size_t row, size_t col) const noexcept;

  /// Matrix operations
  [[nodiscard]] Mat2 operator*(const Mat2& other) const noexcept;
  [[nodiscard]] Mat2 adjoint() const noexcept;
  [[nodiscard]] Mat2 power(double exponent) const;

  /// Numerical queries
  [[nodiscard]] bool isApproxEqual(const Mat2& other,
                                   double tolerance = 1e-14) const noexcept;
  [[nodiscard]] bool isUnitary(double tolerance = 1e-14) const noexcept;
  [[nodiscard]] bool isHermitian(double tolerance = 1e-14) const noexcept;
  [[nodiscard]] bool isDiagonal(double tolerance = 1e-14) const noexcept;
};

} // namespace mqt::unitary
```

**Design Rationale:**

- **Cache-Friendly:** 64-byte alignment fits exactly in one cache line on most architectures
- **constexpr Support:** Common gates can be computed at compile time, eliminating runtime overhead
- **Row-Major Storage:** Matches typical access patterns in matrix operations
- **Numerical Tolerance:** Machine precision (≈10⁻¹⁴) used for floating-point comparisons
- **Qubit-Order Independent:** Single-qubit gates have no ordering ambiguity

#### Mat4: Two-Qubit Matrices

The `Mat4` class represents 4×4 complex matrices for two-qubit gates:

```cpp
namespace mqt::unitary {

/// 4×4 complex matrix for two-qubit gates (stack-allocated, 256 bytes)
///
/// Qubit Ordering Convention:
/// - Qubits are ordered as [q0, q1] where q0 is the LSB (position 0)
/// - Basis states: |00⟩, |01⟩, |10⟩, |11⟩ (binary: 0, 1, 2, 3)
/// - Matrix element M[i][j] represents amplitude from basis state j to state i
class Mat4 {
public:
  using Complex = std::complex<double>;

  /// Stack storage: 16 complex values = 32 doubles = 256 bytes
  /// Fits in four cache lines
  std::array<Complex, 16> data;

  /// Qubit order for this matrix (which qubits, in what positions)
  /// Empty for anonymous matrices; populated for gate-derived matrices
  QubitOrder qubits;

  /// Constructors
  constexpr Mat4() = default;
  explicit constexpr Mat4(std::array<Complex, 16> values);
  explicit Mat4(std::array<Complex, 16> values, QubitOrder order);

  /// Static factory methods for common gates
  /// These assume canonical two-qubit ordering [q0, q1]
  static constexpr Mat4 identity() noexcept;
  static constexpr Mat4 cnot() noexcept;      // Control on q0, target on q1
  static constexpr Mat4 cz() noexcept;        // Control on q0, target on q1
  static constexpr Mat4 swap() noexcept;      // Symmetric: swaps q0 and q1

  /// Element access (row-major ordering)
  constexpr Complex& operator()(size_t row, size_t col) noexcept;
  constexpr const Complex& operator()(size_t row, size_t col) const noexcept;

  /// Matrix operations
  [[nodiscard]] Mat4 operator*(const Mat4& other) const;
  [[nodiscard]] Mat4 adjoint() const noexcept;
  [[nodiscard]] Mat4 power(double exponent) const;

  /// Construction from smaller matrices
  [[nodiscard]] static Mat4 tensorProduct(const Mat2& left, const Mat2& right);
  [[nodiscard]] static Mat4 controlled(const Mat2& target,
                                       bool positiveControl = true);

  /// Qubit reordering
  /// Permute the qubit order of this matrix to match a target ordering
  /// Example: if this matrix is for [q0, q1] and target is [q1, q0],
  /// returns a new matrix with basis reordered accordingly
  [[nodiscard]] Mat4 permuteQubits(const QubitOrder& targetOrder) const;

  /// Apply a permutation directly (permutation[i] = position of target qubit i in source)
  [[nodiscard]] Mat4 applyPermutation(mlir::ArrayRef<size_t> permutation) const;

  /// Numerical queries
  [[nodiscard]] bool isApproxEqual(const Mat4& other,
                                   double tolerance = 1e-14) const noexcept;
  [[nodiscard]] bool isUnitary(double tolerance = 1e-14) const noexcept;
  [[nodiscard]] bool isHermitian(double tolerance = 1e-14) const noexcept;
  [[nodiscard]] bool isDiagonal(double tolerance = 1e-14) const noexcept;
};

} // namespace mqt::unitary
```

**Design Rationale:**

- **Controlled Gate Construction:** Efficient expansion from single-qubit gates to controlled two-qubit gates
- **Tensor Product Support:** Enables construction of product states and parallel gate application
- **Consistent Interface:** Mirrors `Mat2` API for uniform usage patterns
- **Explicit Qubit Tracking:** `qubits` field maintains ordering information for composition correctness
- **Permutation Support:** Efficient basis reordering when composing gates with different qubit orders

#### Permutation Algorithm

The basis permutation algorithm reorders matrix elements according to the qubit permutation:

```cpp
namespace mqt::unitary {

/// Apply a qubit permutation to a 2^n × 2^n matrix
/// permutation[i] specifies the source position of target qubit i
///
/// Algorithm: For each basis state |b⟩:
///   1. Interpret b as n-bit number with qubits in target order
///   2. Remap bits according to permutation to get source state |b'⟩
///   3. Copy M[b'][c'] → M_out[b][c] for all b, c
///
/// Time complexity: O(4^n) for n-qubit gates
/// Space complexity: O(4^n) for output matrix (input unchanged)
template<size_t N>
[[nodiscard]] std::array<std::complex<double>, N * N>
  permuteMatrixBasis(const std::array<std::complex<double>, N * N>& matrix,
                     mlir::ArrayRef<size_t> permutation,
                     size_t numQubits);

} // namespace mqt::unitary
```

**Example:** Consider `CNOT[q0, q1]` (control q0, target q1) and we want `CNOT[q1, q0]`:

- Permutation: `[1, 0]` (target position 0 gets source position 1; target position 1 gets source position 0)
- Basis mapping: `|00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩`
- Result: CNOT with swapped control/target roles

### Symbolic Expression Framework

For gates with dynamic parameters, complex compositions, or deferred evaluation, the library provides a symbolic expression tree system.

#### Base Expression Interface

```cpp
namespace mqt::unitary {

/// Base class for symbolic unitary expressions
class UnitaryExpr {
public:
  virtual ~UnitaryExpr() = default;

  /// Query expression structure
  [[nodiscard]] virtual size_t getNumQubits() const = 0;
  [[nodiscard]] virtual bool canMaterialize() const = 0;
  [[nodiscard]] virtual bool isFullyStatic() const = 0;

  /// Get the qubits this expression operates on (in order)
  [[nodiscard]] virtual QubitOrder getQubitOrder() const = 0;

  /// Materialization (returns std::nullopt if not possible)
  [[nodiscard]] virtual std::optional<Mat2> materializeMat2() const;
  [[nodiscard]] virtual std::optional<Mat4> materializeMat4() const;
  [[nodiscard]] virtual mlir::DenseElementsAttr
    materializeDense(mlir::MLIRContext* ctx) const;

  /// Symbolic property queries
  [[nodiscard]] virtual bool isHermitian() const = 0;
  [[nodiscard]] virtual bool isDiagonal() const = 0;

  /// Smart pointer for ownership
  using Ptr = std::unique_ptr<UnitaryExpr>;
};

} // namespace mqt::unitary
```

**Design Philosophy:**

- **Lazy Evaluation:** Build expression trees without immediate computation
- **Partial Information:** Query properties without full materialization
- **Optimization Opportunities:** Simplify expressions symbolically before materializing

#### Expression Node Types

##### ConstExpr: Concrete Matrices

```cpp
/// Fully materialized matrix expression
class ConstExpr : public UnitaryExpr {
  std::variant<Mat2, Mat4> matrix;
  QubitOrder qubits;

public:
  explicit ConstExpr(Mat2 m);
  explicit ConstExpr(Mat4 m);
  explicit ConstExpr(Mat2 m, QubitOrder order);
  explicit ConstExpr(Mat4 m, QubitOrder order);

  [[nodiscard]] size_t getNumQubits() const override;
  [[nodiscard]] QubitOrder getQubitOrder() const override { return qubits; }
  [[nodiscard]] bool canMaterialize() const override { return true; }
  [[nodiscard]] bool isFullyStatic() const override { return true; }

  [[nodiscard]] std::optional<Mat2> materializeMat2() const override;
  [[nodiscard]] std::optional<Mat4> materializeMat4() const override;

  [[nodiscard]] bool isHermitian() const override;
  [[nodiscard]] bool isDiagonal() const override;
};
```

##### MulExpr: Matrix Composition with Qubit Reordering

```cpp
/// Matrix multiplication (composition of unitaries)
/// Handles automatic qubit reordering when composing gates with different orderings
class MulExpr : public UnitaryExpr {
  UnitaryExpr::Ptr left;
  UnitaryExpr::Ptr right;
  QubitOrder resultOrder;  // Cached result qubit order

public:
  MulExpr(UnitaryExpr::Ptr l, UnitaryExpr::Ptr r);

  [[nodiscard]] size_t getNumQubits() const override;
  [[nodiscard]] QubitOrder getQubitOrder() const override { return resultOrder; }
  [[nodiscard]] bool canMaterialize() const override;
  [[nodiscard]] bool isFullyStatic() const override;

  [[nodiscard]] std::optional<Mat2> materializeMat2() const override;
  [[nodiscard]] std::optional<Mat4> materializeMat4() const override;

  [[nodiscard]] bool isHermitian() const override;
  [[nodiscard]] bool isDiagonal() const override;

private:
  /// Compute the result qubit order from left and right expressions
  /// Uses left's ordering as the canonical target
  static QubitOrder computeResultOrder(const UnitaryExpr& left,
                                       const UnitaryExpr& right);
};
```

**Materialization Strategy with Reordering:**

When materializing a `MulExpr`:

1. Check if both children have the same qubit order → direct composition (fast path)
2. If orders differ, align them:
   - Use `left`'s qubit order as canonical
   - Permute `right`'s matrix to match `left`'s order
   - Compose the aligned matrices
3. Result inherits `left`'s qubit order

**Example:**

```cpp
// Left: CNOT with qubits [q0, q1]
// Right: CZ with qubits [q1, q0] (opposite order)
//
// Composition process:
// 1. Identify qubit order mismatch
// 2. Permute right's matrix: CZ[q1,q0] → CZ[q0,q1]
// 3. Compose: CNOT[q0,q1] * CZ[q0,q1]
// 4. Result has order [q0, q1]
```

##### AdjExpr: Adjoint (Inverse)

```cpp
/// Adjoint (conjugate transpose) of a unitary
class AdjExpr : public UnitaryExpr {
  UnitaryExpr::Ptr child;

public:
  explicit AdjExpr(UnitaryExpr::Ptr c);

  [[nodiscard]] size_t getNumQubits() const override;
  [[nodiscard]] bool canMaterialize() const override;
  [[nodiscard]] bool isFullyStatic() const override;

  [[nodiscard]] std::optional<Mat2> materializeMat2() const override;
  [[nodiscard]] std::optional<Mat4> materializeMat4() const override;

  [[nodiscard]] bool isHermitian() const override;
  [[nodiscard]] bool isDiagonal() const override;
};
```

**Property Propagation:** If child is Hermitian, adjoint equals child; diagonal property preserved.

##### PowExpr: Matrix Exponentiation

```cpp
/// Power of a unitary (U^exponent)
class PowExpr : public UnitaryExpr {
  UnitaryExpr::Ptr child;
  double exponent;

public:
  PowExpr(UnitaryExpr::Ptr c, double exp);

  [[nodiscard]] size_t getNumQubits() const override;
  [[nodiscard]] bool canMaterialize() const override;
  [[nodiscard]] bool isFullyStatic() const override;

  [[nodiscard]] std::optional<Mat2> materializeMat2() const override;
  [[nodiscard]] std::optional<Mat4> materializeMat4() const override;

  [[nodiscard]] bool isHermitian() const override;
  [[nodiscard]] bool isDiagonal() const override;
};
```

**Exponentiation Strategy:**

- **Integer powers:** Repeated multiplication
- **Fractional powers:** Eigendecomposition and diagonal exponentiation
- **Special cases:** Identity for exponent 0, adjoint for exponent -1

##### CtrlExpr: Control Expansion

```cpp
/// Controlled unitary expansion
class CtrlExpr : public UnitaryExpr {
  UnitaryExpr::Ptr target;
  mlir::SmallVector<mlir::Value, 2> posControls;
  mlir::SmallVector<mlir::Value, 2> negControls;
  QubitOrder resultOrder;  // Control qubits followed by target qubits

public:
  CtrlExpr(UnitaryExpr::Ptr t,
           mlir::ArrayRef<mlir::Value> posCtrl,
           mlir::ArrayRef<mlir::Value> negCtrl);

  [[nodiscard]] size_t getNumQubits() const override;
  [[nodiscard]] QubitOrder getQubitOrder() const override { return resultOrder; }
  [[nodiscard]] bool canMaterialize() const override;
  [[nodiscard]] bool isFullyStatic() const override;

  [[nodiscard]] mlir::DenseElementsAttr
    materializeDense(mlir::MLIRContext* ctx) const override;

  [[nodiscard]] bool isHermitian() const override;
  [[nodiscard]] bool isDiagonal() const override;

private:
  /// Compute canonical qubit ordering: controls first, then targets
  static QubitOrder computeResultOrder(mlir::ArrayRef<mlir::Value> posCtrl,
                                       mlir::ArrayRef<mlir::Value> negCtrl,
                                       const UnitaryExpr& target);
};
```

**Qubit Ordering for Controlled Gates:**

- Control qubits appear first in the qubit order (positive controls, then negative controls)
- Target qubits follow controls
- Example: `ctrl(%q2, %q3) { gate %q0, %q1 }` → order is `[q2, q3, q0, q1]`

##### ParamExpr: Parametrized Gates

```cpp
/// Parametrized expression with symbolic parameters
class ParamExpr : public UnitaryExpr {
  mlir::SmallVector<size_t, 4> paramIndices;
  std::function<Mat2(mlir::ArrayRef<double>)> generator;

public:
  ParamExpr(mlir::ArrayRef<size_t> indices,
            std::function<Mat2(mlir::ArrayRef<double>)> gen);

  [[nodiscard]] size_t getNumQubits() const override { return 1; }
  [[nodiscard]] bool canMaterialize() const override { return false; }
  [[nodiscard]] bool isFullyStatic() const override { return false; }

  /// Partial evaluation: given parameter values, materialize
  [[nodiscard]] Mat2 evaluate(mlir::ArrayRef<double> paramValues) const;

  [[nodiscard]] bool isHermitian() const override;
  [[nodiscard]] bool isDiagonal() const override;
};
```

**Design Notes:**

- **Multiple Parameters:** Supports gates with arbitrary parameter counts (e.g., `u(θ, φ, λ)` has three parameters)
- **Parameter Tracking:** Maintains indices for parameter identification across expressions
- **Generator Function:** Callable that produces the matrix given parameter values

## Core Operations and Algorithms

### Composition

Matrix composition represents sequential application of quantum gates (right-to-left convention):

```cpp
namespace mqt::unitary {

/// Compose two unitaries (matrix multiplication: b then a)
/// Automatically handles qubit reordering if necessary
[[nodiscard]] UnitaryExpr::Ptr compose(UnitaryExpr::Ptr a, UnitaryExpr::Ptr b);

/// Fast path for static 2×2 composition (no qubit ordering issues)
[[nodiscard]] Mat2 compose(const Mat2& a, const Mat2& b) noexcept;

/// Fast path for static 4×4 composition with explicit qubit ordering
/// If qubit orders match, performs direct multiplication
/// If orders differ, permutes b's matrix before multiplication
[[nodiscard]] Mat4 compose(const Mat4& a, const Mat4& b);

/// Compose sequence of unitaries (applied left-to-right)
/// Maintains consistent qubit ordering throughout composition
[[nodiscard]] UnitaryExpr::Ptr
  composeSequence(mlir::ArrayRef<UnitaryExpr::Ptr> unitaries);

} // namespace mqt::unitary
```

**Optimization Strategy:**

- **Same-order fast path:** When gates operate on the same qubits in the same order, skip permutation logic entirely
- **Reordering path:** When orders differ, apply permutation then compose
- **Single-qubit gates:** No ordering issues; always use fast path
- **Symbolic expressions:** Build `MulExpr` nodes that defer reordering until materialization

#### Composition Algorithm Details

```cpp
namespace mqt::unitary {

/// Detailed composition algorithm for two 4×4 matrices
///
/// Given: matrices A (qubits [qa0, qa1]) and B (qubits [qb0, qb1])
/// Goal: Compute A * B correctly accounting for qubit ordering
///
/// Cases:
/// 1. Same order ([qa0, qa1] == [qb0, qb1]): Direct multiplication
/// 2. Different order:
///    a. Check if qubits overlap: must have same qubit set
///    b. Compute permutation: π such that qb[π[i]] = qa[i]
///    c. Apply permutation to B: B' = permute(B, π)
///    d. Multiply: A * B'
///    e. Result has A's qubit order
///
[[nodiscard]] Mat4 composeWithReordering(const Mat4& a, const Mat4& b);

} // namespace mqt::unitary
```

**Example Scenario:**

```cpp
// Gate sequence: H⊗H on [q0,q1], then CNOT on [q1,q0], then CZ on [q0,q1]

// Step 1: H⊗H with order [q0, q1]
Mat2 h = Mat2::hadamard();
Mat4 hh = Mat4::tensorProduct(h, h);
hh.qubits = {q0, q1};

// Step 2: CNOT with order [q1, q0] (control=q1, target=q0)
Mat4 cnot = Mat4::cnot();  // Assumes [control, target]
cnot.qubits = {q1, q0};

// Step 3: Compose HH with CNOT
// - HH has order [q0, q1]
// - CNOT has order [q1, q0]
// - Permutation: [1, 0] (swap qubits)
// - Permute CNOT: CNOT' with order [q0, q1]
// - Result: HH * CNOT' with order [q0, q1]
Mat4 intermediate = compose(hh, cnot);

// Step 4: CZ with order [q0, q1]
Mat4 cz = Mat4::cz();
cz.qubits = {q0, q1};

// Step 5: Compose with CZ
// - intermediate has order [q0, q1]
// - CZ has order [q0, q1]
// - Same order → direct multiplication (fast path)
Mat4 result = compose(intermediate, cz);
```

### Mixed Single-Qubit and Multi-Qubit Composition

A particularly important case is composing sequences where single-qubit and two-qubit gates are interleaved:

```cpp
namespace mqt::unitary {

/// Compose a single-qubit gate into a two-qubit context
///
/// Given: 2×2 matrix G for single-qubit gate on qubit q
///        Qubit context [q0, q1] for the two-qubit space
/// Result: 4×4 matrix representing G ⊗ I or I ⊗ G depending on which qubit
///
/// Example: X gate on q1 in context [q0, q1] → I ⊗ X
///          X gate on q0 in context [q0, q1] → X ⊗ I
///
[[nodiscard]] Mat4 embedSingleQubit(const Mat2& gate,
                                    mlir::Value targetQubit,
                                    const QubitOrder& context);

/// Compose a sequence with mixed single and two-qubit gates
/// Automatically expands single-qubit gates to the appropriate two-qubit space
[[nodiscard]] UnitaryExpr::Ptr
  composeMixedSequence(mlir::ArrayRef<UnitaryExpr::Ptr> gates,
                       const QubitOrder& targetContext);

} // namespace mqt::unitary
```

**Example:**

```mlir
// Sequence: CNOT %q0, %q1; H %q0; CZ %q0, %q1
// Context: [q0, q1] (two-qubit space [q0, q1])
//
// Step 1: CNOT on [q0, q1] → 4×4 matrix
// Step 2: H on q0 → expand to (H ⊗ I) in context [q0, q1] → 4×4 matrix
// Step 3: CZ on [q0, q1] → 4×4 matrix
// Step 4: Compose all three 4×4 matrices with consistent [q0, q1] ordering
```

### Example 6: Symbolic Expression with Reordering

Build symbolic expression trees that track qubit ordering:

```cpp
using namespace mqt::unitary;

// Build expression: CNOT[q0,q1] · (CZ[q1,q0])†

// CNOT with order [q0, q1]
Mat4 cnot = Mat4::cnot();
cnot.qubits = {q0, q1};
auto cnotExpr = std::make_unique<ConstExpr>(cnot, QubitOrder{q0, q1});

// CZ with order [q1, q0]
Mat4 cz = Mat4::cz();
cz.qubits = {q1, q0};
auto czExpr = std::make_unique<ConstExpr>(cz, QubitOrder{q1, q0});

// Adjoint of CZ
auto czAdj = std::make_unique<AdjExpr>(std::move(czExpr));

// Compose (handles reordering automatically)
auto composed = std::make_unique<MulExpr>(std::move(cnotExpr),
                                          std::move(czAdj));

// Query result order (inherits from left operand)
QubitOrder resultOrder = composed->getQubitOrder();
assert(resultOrder[0] == q0 && resultOrder[1] == q1);

// Materialize when needed
if (auto result = composed->materializeMat4()) {
  // Matrix is correctly computed with [q0, q1] ordering
  assert(result->qubits[0] == q0);
  assert(result->qubits[1] == q1);
}
```

### Example 7: Parametrized Gates

Handle gates with multiple parameters:

```cpp
using namespace mqt::unitary;

// Generator for U gate with three parameters: u(θ, φ, λ)
auto uGenerator = [](mlir::ArrayRef<double> params) -> Mat2 {
  assert(params.size() == 3);
  double theta = params[0];
  double phi = params[1];
  double lambda = params[2];

  // Matrix for u(θ, φ, λ)
  // [[cos(θ/2), -exp(iλ)sin(θ/2)],
  //  [exp(iφ)sin(θ/2), exp(i(φ+λ))cos(θ/2)]]
  // ... implementation ...
};

// Create parametrized expression
mlir::SmallVector<size_t, 3> paramIndices = {0, 1, 2};
auto uExpr = std::make_unique<ParamExpr>(paramIndices, uGenerator);

// Evaluate with concrete values
mlir::SmallVector<double, 3> values = {M_PI / 2, 0.0, M_PI};
Mat2 concrete = uExpr->evaluate(values);
```

## Testing Strategy

### Unit Tests

**Correctness Tests:**

- Mathematical properties: unitarity (U†U = I), adjoint correctness, power consistency
- Gate identities: Pauli relations, Hadamard properties, rotation periodicity
- Composition associativity: (A·B)·C = A·(B·C)
- Control expansion: verify controlled-unitary structure
- **Qubit ordering:** Verify different qubit orderings produce different matrices
- **Reordering correctness:** Compose A·B with explicit reordering matches permuted composition
- **Mixed sequences:** Single-qubit gates embedded correctly in multi-qubit contexts

**Precision Tests:**

- Machine precision validation: ≈10⁻¹⁴ relative error
- Numerical stability: condition number analysis for matrix operations
- Edge cases: identity matrices, diagonal matrices, Hermitian matrices
- **Permutation accuracy:** Basis reordering preserves matrix unitarity and determinant

### Performance Tests

**Microbenchmarks:**

- Individual operation latencies (multiplication, adjoint, power)
- **Same-order vs different-order composition:** Measure fast-path effectiveness
- **Qubit order detection:** Measure comparison overhead
- **Permutation cost:** Measure basis reordering overhead
- Comparison against reference implementations
- Regression detection across compiler versions

**Allocation Tracking:**

- Verify zero heap allocations for fixed-size operations
- Profile expression tree memory usage
- Detect allocation regressions

**Cache Behavior:**

- Profile L1/L2/L3 cache hit rates
- Measure memory bandwidth utilization
- Optimize for cache line alignment

### Integration Tests

**MLIR Interoperability:**

- Round-trip conversion: Mat2 ↔ DenseElementsAttr
- Attribute preservation across transformations
- Context lifetime management
- **Qubit order preservation:** Verify SSA values maintained through conversions

**Interface Usage:**

- `UnitaryOpInterface` implementation correctness
- Modifier chain materialization with qubit tracking
- Pass infrastructure integration
- **Gate fusion:** Verify qubit order compatibility checks in fusion passes

## Implementation Structure

### Dependencies

The library relies exclusively on LLVM/MLIR infrastructure and C++20 standard library:

- **C++20 Standard Library:**
  - `<complex>`: Complex number arithmetic
  - `<array>`: Fixed-size containers
  - `<optional>`: Optional return types
  - `<variant>`: Type-safe unions
  - `<functional>`: Function objects
  - `<concepts>`: Type constraints
  - `<algorithm>`: Standard algorithms (std::equal, std::find)

- **LLVM ADT:**
  - `llvm/ADT/ArrayRef.h`: Non-owning array reference
  - `llvm/ADT/SmallVector.h`: Small-size optimized vector
  - `llvm/ADT/APFloat.h`: Arbitrary precision floating point

- **MLIR Infrastructure:**
  - `mlir/IR/Attributes.h`: MLIR attribute system
  - `mlir/IR/Builders.h`: Attribute construction utilities
  - `mlir/IR/BuiltinTypes.h`: Tensor and complex types
  - `mlir/IR/MLIRContext.h`: MLIR context management

### Header Organization

```
include/mqt/unitary/
  Mat2.h              // 2×2 matrix class
  Mat4.h              // 4×4 matrix class
  QubitOrder.h        // Qubit ordering utilities
  Permutation.h       // Basis permutation algorithms
  UnitaryExpr.h       // Symbolic expression framework
  Operations.h        // Composition, adjoint, power, control
  Conversion.h        // MLIR attribute interop
  Numerical.h         // Eigendecomposition, exponentiation
  Utils.h             // Numerical utilities, tolerances
```

### Namespace Structure

```cpp
namespace mqt::unitary {
  // Core matrix types
  class Mat2;
  class Mat4;

  // Qubit ordering
  using QubitOrder = mlir::SmallVector<mlir::Value, 4>;
  [[nodiscard]] bool isSameOrder(const QubitOrder&, const QubitOrder&);
  [[nodiscard]] bool isSameQubits(const QubitOrder&, const QubitOrder&);
  [[nodiscard]] std::optional<mlir::SmallVector<size_t, 4>>
    computePermutation(const QubitOrder&, const QubitOrder&);

  // Symbolic expressions
  class UnitaryExpr;
  class ConstExpr;
  class MulExpr;
  class AdjExpr;
  class PowExpr;
  class CtrlExpr;
  class ParamExpr;

  // Operations
  [[nodiscard]] UnitaryExpr::Ptr compose(UnitaryExpr::Ptr, UnitaryExpr::Ptr);
  [[nodiscard]] UnitaryExpr::Ptr adjoint(UnitaryExpr::Ptr);
  [[nodiscard]] UnitaryExpr::Ptr power(UnitaryExpr::Ptr, double);
  [[nodiscard]] Mat4 controlled(const Mat2&, mlir::Value, mlir::Value, bool);

  // Mixed composition
  [[nodiscard]] Mat4 embedSingleQubit(const Mat2&, mlir::Value, const QubitOrder&);
  [[nodiscard]] UnitaryExpr::Ptr
    composeMixedSequence(mlir::ArrayRef<UnitaryExpr::Ptr>, const QubitOrder&);

  // Materialization
  [[nodiscard]] bool canMaterialize(const UnitaryExpr&);
  [[nodiscard]] std::optional<Mat2> materializeMat2(const UnitaryExpr&);
  [[nodiscard]] std::optional<Mat4> materializeMat4(const UnitaryExpr&);
  [[nodiscard]] mlir::DenseElementsAttr materializeAttr(const UnitaryExpr&,
                                                        mlir::MLIRContext*);

  // MLIR conversion
  [[nodiscard]] mlir::DenseElementsAttr toAttr(const Mat2&, mlir::MLIRContext*);
  [[nodiscard]] mlir::DenseElementsAttr toAttr(const Mat4&, mlir::MLIRContext*);
  [[nodiscard]] std::optional<Mat2> fromAttr(mlir::DenseElementsAttr);
  [[nodiscard]] std::optional<Mat4> fromAttr(mlir::DenseElementsAttr);
}
```

## Relationship to Quantum Dialect

The `UnitaryExpr` library provides the computational substrate for the unified unitary interface defined in the MQT quantum dialects (Quartz and Flux):

**Integration Points:**

1. **Matrix Extraction:** Operations implementing `UnitaryOpInterface::tryGetStaticMatrix()` use this library to compute concrete matrices with correct qubit ordering
2. **Symbolic Composition:** Modifier operations (`ctrl`, `inv`, `pow`) build `UnitaryExpr` trees that track qubit ordering through transformations
3. **Equivalence Checking:** Optimization passes use matrix comparison to identify equivalent gate sequences, accounting for qubit order differences
4. **Gate Fusion:** Canonicalization passes compose matrices with automatic reordering to detect fusion opportunities across different qubit orderings

**Design Separation:**

The library maintains clean separation from MLIR dialect definitions:

- **No dialect dependencies:** Can be used standalone for matrix computations
- **Generic interface:** Not tied to specific operation types
- **Reusable infrastructure:** Applicable to future dialects or external tools
- **Qubit-order agnostic core:** Matrix operations work with any ordering convention

This architectural choice enables the library to serve as a foundational component across the entire MQT MLIR infrastructure while maintaining modularity and testability. The explicit handling of qubit ordering ensures correctness when composing gates from different sources or with different qubit arrangements, which is essential for robust optimization passes.
