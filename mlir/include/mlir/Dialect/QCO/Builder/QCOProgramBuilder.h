/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/CBit/IR/CBitAttributes.h"

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;
class Operation;
class ValueRange;

namespace qco {

/// Builder API for constructing quantum programs in the QCO dialect
///
/// The QCOProgramBuilder provides a type-safe interface for constructing
/// quantum circuits using value semantics. Operations consume input qubit
/// SSA values and produce new output values, following the functional
/// programming paradigm.
///
/// @par Linear Type Enforcement:
/// The builder enforces linear type semantics by tracking valid qubit SSA
/// values. Once a qubit is consumed by an operation producing a new version
/// (e.g., reset, measure), the old SSA value is invalidated. This prevents
/// use-after-consume errors and mirrors quantum computing's no-cloning theorem.
///
/// @par Qubit addressing:
/// A program must use either static qubits (`staticQubit`) or dynamic
/// allocation
/// (`allocQubit`, `allocQubitRegister`, or `qtensorAlloc`), never both. The
/// builder terminates with a usage error if the modes are mixed.
///
/// @par Example Usage:
/// ```c++
/// QCOProgramBuilder builder(context);
/// builder.initialize();
///
/// auto q0 = builder.staticQubit(0);
/// auto q1 = builder.staticQubit(1);
///
/// // Operations return updated values
/// q0 = builder.h(q0);
/// std::tie(q0, q1) = builder.cx(q0, q1);
///
/// auto moduleOp = builder.finalize();
/// ```
class QCOProgramBuilder final : public ImplicitLocOpBuilder {
public:
  /// Construct a new QCOProgramBuilder
  /// @param context The MLIR context to use for building operations
  explicit QCOProgramBuilder(MLIRContext* context);

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  /// Initialize the builder and prepare for program construction, with
  /// a default return type of i64.
  ///
  /// Creates a main function with an `mqt.entry_point` attribute. Must be
  /// called before adding operations.
  void initialize();

  /// Initialize the builder and prepare for program construction
  /// with specified return types.
  /// @param returnTypes The return types for the main function
  ///
  /// Creates a main function with an `mqt.entry_point` attribute. Must be
  /// called before adding operations.
  void initialize(TypeRange returnTypes);

  /// Modify the return types of the main function after initialization.
  /// @param returnTypes The new return types for the main function
  void retype(TypeRange returnTypes);

  //===--------------------------------------------------------------------===//
  // Functions
  //===--------------------------------------------------------------------===//

  /// Create a private function.
  ///
  /// The callback must return one trailing qubit for every qubit argument, in
  /// qubit-argument order.
  func::FuncOp
  createFunction(StringRef name, TypeRange argumentTypes,
                 function_ref<SmallVector<Value>(ValueRange)> body);

  /// Create a complete private unitary function.
  func::FuncOp
  createUnitaryFunction(StringRef name, TypeRange argumentTypes,
                        function_ref<SmallVector<Value>(ValueRange)> body);

  /// Call a function, using `qco.call` for a unitary function.
  ///
  /// Ordinary results are followed by the updated qubit arguments.
  SmallVector<Value> call(func::FuncOp callee, ValueRange operands);

  //===--------------------------------------------------------------------===//
  // Constants
  //===--------------------------------------------------------------------===//

  /// Create a constant integer value
  /// @param value The value to store in the constant
  /// @return The value produced by the constant operation
  ///
  /// @par Example:
  /// ```c++
  /// auto c = builder.intConstant(1);
  /// ```
  /// ```mlir
  /// %c = arith.constant 1 : i64
  /// ```
  Value intConstant(int64_t value);

  /// Create a constant float value
  /// @param value The value to store in the constant
  /// @return The value produced by the constant operation
  ///
  /// @par Example:
  /// ```c++
  /// auto c = builder.floatConstant(0.123);
  /// ```
  /// ```mlir
  /// %c = arith.constant 0.123 : f64
  /// ```
  Value floatConstant(double value);

  /// Create a constant boolean value
  /// @param value The value to store in the constant
  /// @return The value produced by the constant operation
  ///
  /// @par Example:
  /// ```c++
  /// auto c = builder.boolConstant(true);
  /// ```
  /// ```mlir
  /// %c = arith.constant 1 : i1
  /// ```
  Value boolConstant(bool value);

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// A tracked qubit value and its register information.
  struct Qubit {
    /// The tracked SSA value
    Value value;
    /// ID of the register the qubit belongs to
    int64_t regId = -1;
    /// Index of the qubit within its register
    Value regIndex;

    /// Implicitly construct a tracked qubit from an SSA value.
    /// @param value The underlying qubit SSA value
    /// @param regId ID of the register containing the qubit, or `-1`
    /// @param regIndex Index of the qubit within its register, if applicable
    // NOLINTNEXTLINE(google-explicit-constructor)
    Qubit(Value value, int64_t regId = -1, Value regIndex = {})
        : value(value), regId(regId), regIndex(regIndex) {}

    /// Implicitly convert this tracked qubit to its underlying SSA value.
    /// @return The underlying `Value`
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator Value() const { return value; }

    /// Get the type of the underlying SSA value.
    /// @return The underlying value's type
    Type getType() const { return value.getType(); }

    /// Get the operation defining the underlying SSA value.
    /// @return The defining operation, or `nullptr` if the value has none
    Operation* getDefiningOp() const { return value.getDefiningOp(); }

    /// Get the operation defining the underlying SSA value as @p OpTy.
    /// @tparam OpTy The expected defining operation type
    /// @return The defining operation as @p OpTy, or a null operation if the
    /// value has no defining operation or it is not of type @p OpTy
    template <typename OpTy> OpTy getDefiningOp() const {
      return value.getDefiningOp<OpTy>();
    }
  };

  /// A tracked qubit tensor value and its register information.
  struct Tensor {
    /// The tracked SSA value
    Value value;
    /// ID of the register the tensor corresponds to
    int64_t regId = -1;

    /// Implicitly construct a tracked tensor from an SSA value.
    /// @param value The underlying tensor SSA value
    /// @param regId ID of the corresponding register, or `-1`
    // NOLINTNEXTLINE(google-explicit-constructor)
    Tensor(Value value, int64_t regId = -1) : value(value), regId(regId) {}

    /// Implicitly convert this tracked tensor to its underlying SSA
    /// value.
    /// @return The underlying `Value`
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator Value() const { return value; }

    /// Get the type of the underlying SSA value.
    /// @return The underlying value's type
    Type getType() const { return value.getType(); }

    /// Get the operation defining the underlying SSA value.
    /// @return The defining operation, or `nullptr` if the value has none
    Operation* getDefiningOp() const { return value.getDefiningOp(); }

    /// Get the operation defining the underlying SSA value as @p OpTy.
    /// @tparam OpTy The expected defining operation type
    /// @return The defining operation as @p OpTy, or a null operation if the
    /// value has no defining operation or it is not of type @p OpTy
    template <typename OpTy> OpTy getDefiningOp() const {
      return value.getDefiningOp<OpTy>();
    }
  };

  /// Represents a qubit register with its qubits.
  struct QubitRegister {
    /// The QTensor value representing the qubit register
    Value value;
    /// The allocated qubit values
    SmallVector<Value> qubits;

    /// Access a specific qubit in the register
    /// @param index The index of the qubit to access
    /// @return The specified qubit value
    Value& operator[](size_t index);

    /// Conversion to the backing QTensor value
    /// @return The QTensor value representing the qubit register
    explicit operator Value() const { return value; }
  };

  /// Allocate a single qubit initialized to |0⟩
  /// @return A tracked qubit handle (convertible to `Value`)
  ///
  /// @par Example:
  /// ```c++
  /// auto q = builder.allocQubit();
  /// ```
  /// ```mlir
  /// %q = qco.alloc : !qco.qubit
  /// ```
  Qubit allocQubit();

  /// Get a static qubit by index
  /// @param index The qubit index
  /// @return A tracked qubit handle (convertible to `Value`)
  ///
  /// @par Example:
  /// ```c++
  /// auto q0 = builder.staticQubit(0);
  /// ```
  /// ```mlir
  /// %q0 = qco.static 0 : !qco.qubit
  /// ```
  Qubit staticQubit(uint64_t index);

  /// Allocate a qubit tensor and eagerly extract every element
  /// @param size Number of qubits (must be positive)
  /// @param name Optional source-level register name
  /// @return A `QubitRegister` containing the residual tensor and one
  /// standalone qubit value for every eagerly extracted element
  ///
  /// @par Example:
  /// ```c++
  /// auto q = builder.allocQubitRegister(3);
  /// ```
  /// ```mlir
  /// %t0 = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
  /// %t1, %q0 = qtensor.extract %t0[%c0]: tensor<3x!qco.qubit>
  /// %t2, %q1 = qtensor.extract %t1[%c1]: tensor<3x!qco.qubit>
  /// %t3, %q2 = qtensor.extract %t2[%c2]: tensor<3x!qco.qubit>
  /// ```
  QubitRegister allocQubitRegister(int64_t size, StringRef name = {});

  /// Allocate a classical bit register
  ///
  /// The register uses `!cbit.reg<N>`. Its initialization is explicit
  /// and independent of every other register built by this builder.
  ///
  /// @param size Number of bits (must be positive)
  /// @param name Optional source-level register name; defaults to no name
  /// @param initialization Initial value of the register elements; defaults to
  /// zero
  /// @return The CBit register value
  ///
  /// @par Example:
  /// ```c++
  /// auto c = builder.allocClassicalBitRegister(3, "c");
  /// ```
  /// ```mlir
  /// %c = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"}
  ///     : !cbit.reg<3>
  /// ```
  Value allocClassicalBitRegister(
      int64_t size, StringRef name = {},
      cbit::Initialization initialization = cbit::Initialization::Zero);

  /// Load one value from a classical-bit register.
  Value loadClassicalBit(Value reg, const std::variant<int64_t, Value>& index);

  /// Store one value in a classical-bit register.
  void storeClassicalBit(Value value, Value reg,
                         const std::variant<int64_t, Value>& index);

  //===--------------------------------------------------------------------===//
  // QTensor operations
  //===--------------------------------------------------------------------===//

  /// Allocate a qubit tensor
  ///
  /// Allocates and returns one intact, one-dimensional tensor of
  /// `!qco.qubit` values. No elements are extracted. If the size is a constant,
  /// the tensor has static size; otherwise it has dynamic size. Its qubits are
  /// initialized in the |0> state, and the tensor is tracked automatically.
  ///
  /// @param size Number of qubits (must be positive)
  /// @return The allocated tensor
  ///
  /// @par Example:
  /// ```c++
  /// auto tensor = builder.qtensorAlloc(3);
  /// ```
  /// ```mlir
  /// %tensor = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
  /// ```
  Value qtensorAlloc(const std::variant<int64_t, Value>& size);

  /// Allocate a qubit tensor from a list of qubit values
  ///
  /// Consumes the input qubits and creates a one-dimensional tensor of
  /// !qco.qubit types. The resulting tensor has a static size given by the
  /// number of input values. The consumed qubits are removed from the qubit
  /// tracking and the resulting tensor is added to the tracking.
  ///
  /// @param elements Inserted Qubits (must be valid/unconsumed)
  /// @return The allocated tensor
  ///
  /// @par Example:
  /// ```c++
  /// auto tensor = builder.qtensorFromElements({q0, q1, q2});
  /// ```
  /// ```mlir
  /// %tensor = qtensor.from_elements %q0, %q1, %q2 : tensor<3x!qco.qubit>
  /// ```
  Value qtensorFromElements(ValueRange elements);

  /// Extract a qubit from a tensor
  ///
  /// Extracts a qubit from a one-dimensional tensor of qubits at the given
  /// index and returns the updated tensor and the extracted qubit. The
  /// extracted qubit is added to the qubit tracking and the tracking of the
  /// source tensor is updated.
  ///
  /// @param tensor Source tensor (must be valid/unconsumed)
  /// @param index The index from where the qubit is extracted
  /// @return Pair of (outTensor, extractedQubit)
  ///
  /// @par Example:
  /// ```c++
  /// auto [outTensor, q0] = builder.qtensorExtract(tensor, 0);
  /// ```
  /// ```mlir
  /// %outTensor, %q0 = qtensor.extract %tensor[%c0]: tensor<3x!qco.qubit>
  /// ```
  std::pair<Value, Value>
  qtensorExtract(Value tensor, const std::variant<int64_t, Value>& index);

  /// Insert a qubit into a tensor
  ///
  /// Inserts a scalar qubit into the one-dimensional tensor of qubits at the
  /// given index. The inserted qubit is consumed and removed from the qubit
  /// tracking while the tracking for the source tensor is updated.
  ///
  /// @param scalar The scalar qubit that is inserted (must be valid/unconsumed)
  /// @param tensor The tensor where the qubit is inserted (must be
  /// valid/unconsumed)
  /// @param index The index into where the qubit is inserted
  /// @return The output tensor
  ///
  /// @par Example:
  /// ```c++
  /// auto outTensor = builder.qtensorInsert(q0, tensor, 0);
  /// ```
  /// ```mlir
  /// %outTensor = qtensor.insert %q0 into %tensor[%c0] : tensor<3x!qco.qubit>
  /// ```
  Value qtensorInsert(Value scalar, Value tensor,
                      const std::variant<int64_t, Value>& index);

  /// Explicitly deallocate a tensor
  ///
  /// Validates and removes the tensor from tracking. Qubits or tensors of
  /// qubits that were extracted from the tensor but not inserted back again
  /// need to be deallocated separately. Optional; `finalize()` automatically
  /// deallocates all remaining tensors.
  ///
  /// @param tensor Tensor to deallocate (must be valid/unconsumed)
  /// @return Reference to this builder for method chaining
  ///
  /// @par Example:
  /// ```c++
  /// builder.qtensorDealloc(tensor);
  /// ```
  /// ```mlir
  /// qtensor.dealloc %tensor : tensor<3x!qco.qubit>
  /// ```
  QCOProgramBuilder& qtensorDealloc(Value tensor);

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /// Measure a qubit in the computational basis
  ///
  /// Consumes the input qubit and produces a new output qubit SSA value
  /// along with the measurement result (i1). The input is validated and
  /// tracking is updated to reflect the new output value.
  ///
  /// @param qubit Input qubit (must be valid/unconsumed)
  /// @return Pair of (output_qubit, measurement_result)
  ///
  /// @par Example:
  /// ```c++
  /// auto [q_out, result] = builder.measure(q);
  /// ```
  /// ```mlir
  /// %q_out, %result = qco.measure %q : !qco.qubit
  /// ```
  std::pair<Value, Value> measure(Value qubit);

  /// Measure a qubit and store the result in a classical bit register
  ///
  /// Measures the qubit and stores the classical result in the given classical
  /// register at the given index, in addition to returning it.
  ///
  /// @param qubit Input qubit (must be valid/unconsumed)
  /// @param reg The CBit register
  /// @param index The index within the classical register
  /// @return Pair of (output_qubit, measurement_result)
  ///
  /// @par Example:
  /// ```c++
  /// auto [q0_out, r0] = builder.measure(q0, c, 0);
  /// ```
  /// ```mlir
  /// %q0_out, %r0 = qco.measure %q0 : !qco.qubit
  /// cbit.store %r0, %c[%c0] : !cbit.reg<3>
  /// ```
  std::pair<Value, Value> measure(Value qubit, Value reg,
                                  const std::variant<int64_t, Value>& index);

  /// Reset a qubit to |0⟩ state
  ///
  /// Consumes the input qubit and produces a new output qubit SSA value
  /// in the |0⟩ state. The input is validated and tracking is updated.
  ///
  /// @param qubit Input qubit (must be valid/unconsumed)
  /// @return Output qubit value
  ///
  /// @par Example:
  /// ```c++
  /// q = builder.reset(q);
  /// ```
  /// ```mlir
  /// %q_out = qco.reset %q : !qco.qubit -> !qco.qubit
  /// ```
  Value reset(Value qubit);

  //===--------------------------------------------------------------------===//
  // Unitary Operations
  //===--------------------------------------------------------------------===//

  // ZeroTargetOneParameter

#define DECLARE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)            \
  void OP_NAME(const std::variant<double, Value>&(PARAM));                     \
  Value c##OP_NAME(const std::variant<double, Value>&(PARAM), Value control);  \
  ValueRange mc##OP_NAME(const std::variant<double, Value>&(PARAM),            \
                         ValueRange controls);

  DECLARE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DECLARE_ZERO_TARGET_ONE_PARAMETER

  // OneTargetZeroParameter

#define DECLARE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                   \
  Value OP_NAME(Value qubit);                                                  \
  std::pair<Value, Value> c##OP_NAME(Value control, Value target);             \
  std::pair<ValueRange, Value> mc##OP_NAME(ValueRange controls, Value target);

  DECLARE_ONE_TARGET_ZERO_PARAMETER(IdOp, id)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(XOp, x)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(YOp, y)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(ZOp, z)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(HOp, h)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(SOp, s)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(SdgOp, sdg)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(TOp, t)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(TdgOp, tdg)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(SXOp, sx)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, sxdg)

#undef DECLARE_ONE_TARGET_ZERO_PARAMETER

  // OneTargetOneParameter

#define DECLARE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  Value OP_NAME(const std::variant<double, Value>&(PARAM), Value qubit);       \
  std::pair<Value, Value> c##OP_NAME(                                          \
      const std::variant<double, Value>&(PARAM), Value control, Value target); \
  std::pair<ValueRange, Value> mc##OP_NAME(                                    \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value target);

  DECLARE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(POp, p, theta)

#undef DECLARE_ONE_TARGET_ONE_PARAMETER

  // OneTargetTwoParameter

#define DECLARE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)    \
  Value OP_NAME(const std::variant<double, Value>&(PARAM1),                    \
                const std::variant<double, Value>&(PARAM2), Value qubit);      \
  std::pair<Value, Value> c##OP_NAME(                                          \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target);                                                           \
  std::pair<ValueRange, Value> mc##OP_NAME(                                    \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target);

  DECLARE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
  DECLARE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DECLARE_ONE_TARGET_TWO_PARAMETER

  // OneTargetThreeParameter

#define DECLARE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,  \
                                           PARAM3)                             \
  Value OP_NAME(const std::variant<double, Value>&(PARAM1),                    \
                const std::variant<double, Value>&(PARAM2),                    \
                const std::variant<double, Value>&(PARAM3), Value qubit);      \
  std::pair<Value, Value> c##OP_NAME(                                          \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target);                                                           \
  std::pair<ValueRange, Value> mc##OP_NAME(                                    \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target);

  DECLARE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DECLARE_ONE_TARGET_THREE_PARAMETER

  // TwoTargetZeroParameter

#define DECLARE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                   \
  std::pair<Value, Value> OP_NAME(Value qubit0, Value qubit1);                 \
  std::pair<Value, std::pair<Value, Value>> c##OP_NAME(                        \
      Value control, Value qubit0, Value qubit1);                              \
  std::pair<ValueRange, std::pair<Value, Value>> mc##OP_NAME(                  \
      ValueRange controls, Value qubit0, Value qubit1);

  DECLARE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DECLARE_TWO_TARGET_ZERO_PARAMETER

  // TwoTargetOneParameter

#define DECLARE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  std::pair<Value, Value> OP_NAME(const std::variant<double, Value>&(PARAM),   \
                                  Value qubit0, Value qubit1);                 \
  std::pair<Value, std::pair<Value, Value>> c##OP_NAME(                        \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1);                                                           \
  std::pair<ValueRange, std::pair<Value, Value>> mc##OP_NAME(                  \
      const std::variant<double, Value>&(PARAM), ValueRange controls,          \
      Value qubit0, Value qubit1);

  DECLARE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DECLARE_TWO_TARGET_ONE_PARAMETER

  // TwoTargetTwoParameter

#define DECLARE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)    \
  std::pair<Value, Value> OP_NAME(const std::variant<double, Value>&(PARAM1),  \
                                  const std::variant<double, Value>&(PARAM2),  \
                                  Value qubit0, Value qubit1);                 \
  std::pair<Value, std::pair<Value, Value>> c##OP_NAME(                        \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1);                                                           \
  std::pair<ValueRange, std::pair<Value, Value>> mc##OP_NAME(                  \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value qubit0, Value qubit1);

  DECLARE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
  DECLARE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DECLARE_TWO_TARGET_TWO_PARAMETER

  // ThreeTargetZeroParameter

#define DECLARE_THREE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                 \
  std::tuple<Value, Value, Value> OP_NAME(Value qubit0, Value qubit1,          \
                                          Value qubit2);                       \
  std::pair<Value, std::tuple<Value, Value, Value>> c##OP_NAME(                \
      Value control, Value qubit0, Value qubit1, Value qubit2);                \
  std::pair<ValueRange, std::tuple<Value, Value, Value>> mc##OP_NAME(          \
      ValueRange controls, Value qubit0, Value qubit1, Value qubit2);

  DECLARE_THREE_TARGET_ZERO_PARAMETER(RCCXOp, rccx)

#undef DECLARE_THREE_TARGET_ZERO_PARAMETER

  // BarrierOp

  /// Apply a BarrierOp
  ///
  /// @param qubits Input qubits (must be valid/unconsumed)
  /// @return Output qubits
  ///
  /// @par Example:
  /// ```c++
  /// builder.barrier({q0, q1});
  /// ```
  /// ```mlir
  /// qco.barrier %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit,
  /// !qco.qubit
  /// ```
  ValueRange barrier(ValueRange qubits);

  /// Apply an explicitly represented dense unitary matrix
  ///
  /// @param qubits Input qubits (must be valid/unconsumed), ordered from the
  /// most-significant basis bit to the least-significant basis bit
  /// @param matrix Square row-major `complex<f64>` matrix
  /// @return Output qubits
  ValueRange unitary(ValueRange qubits, DenseElementsAttr matrix);

  //===--------------------------------------------------------------------===//
  // Modifiers
  //===--------------------------------------------------------------------===//

  /// Apply a control modifier to a collection of gates
  ///
  /// @param controls Input control qubits
  /// @param targets Input target qubits
  /// @param body Function that builds the body containing the target gates
  /// @return Pair of (output_control_qubits, output_target_qubits)
  ///
  /// @par Example:
  /// ```c++
  /// auto [controls_out, targets_out] =
  ///   builder.ctrl(q0_in, q1_in,
  ///     [&](ValueRange targets) -> SmallVector<Value> {
  ///       return {builder.x(targets[0])};
  ///   });
  /// ```
  /// ```mlir
  /// %controls_out, %targets_out = qco.ctrl(%q0_in) targets(%t = %q1_in) {
  ///   %q1_res = qco.x %t : !qco.qubit -> !qco.qubit
  ///   qco.yield %q1_res : !qco.qubit
  /// } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
  /// ```
  std::pair<ValueRange, ValueRange>
  ctrl(ValueRange controls, ValueRange targets,
       function_ref<SmallVector<Value>(ValueRange)> body);

  /// Apply a control modifier with a single target and one-qubit body.
  ///
  /// @param controls Control qubits
  /// @param target Target qubit
  /// @param body Function that builds the body containing the target operation
  /// @return Pair of (output_control_qubits, output_target_qubit)
  ///
  /// @par Example:
  /// ```c++
  /// auto [controls_out, target_out] =
  ///   builder.ctrl({q0_in, q1_in}, q2_in, [&](Value target) {
  ///     return builder.x(target);
  ///   });
  /// ```
  std::pair<ValueRange, Value> ctrl(ValueRange controls, Value target,
                                    function_ref<Value(Value)> body);

  /// Apply a control modifier with one control and one target.
  ///
  /// @param control Control qubit
  /// @param target Target qubit
  /// @param body Function that builds the body containing the target operation
  /// @return Pair of (output_control_qubit, output_target_qubit)
  ///
  /// @par Example:
  /// ```c++
  /// auto [control_out, target_out] =
  ///   builder.ctrl(q0_in, q1_in, [&](Value target) {
  ///     return builder.x(target);
  ///   });
  /// ```
  std::pair<Value, Value> ctrl(Value control, Value target,
                               function_ref<Value(Value)> body);

  /// Apply an inverse (i.e., adjoint) modifier to a collection of gates
  ///
  /// @param qubits Input qubits
  /// @param body Function that builds the body containing the gates to invert
  /// @return Output qubits
  ///
  /// @par Example:
  /// ```c++
  /// auto qubits_out = builder.inv(q0_in,
  ///   [&](ValueRange qubits) -> SmallVector<Value> {
  ///     return {builder.s(qubits[0])};
  ///   }
  /// );
  /// ```
  /// ```mlir
  /// %qubits_out = qco.inv (%q = %q0_in) {
  ///   %q_res = qco.s %q : !qco.qubit -> !qco.qubit
  ///   qco.yield %q_res : !qco.qubit
  /// } : {!qco.qubit} -> {!qco.qubit}
  /// ```
  ValueRange inv(ValueRange qubits,
                 function_ref<SmallVector<Value>(ValueRange)> body);

  /// Apply an inverse modifier on a single qubit.
  ///
  /// @param qubit Qubit involved in the operation
  /// @param body Function that builds the body containing the operation to
  /// invert
  /// @return Output qubit
  ///
  /// @par Example:
  /// ```c++
  /// auto qubit_out = builder.inv(q0_in, [&](Value qubit) {
  ///   return builder.s(qubit);
  /// });
  /// ```
  Value inv(Value qubit, function_ref<Value(Value)> body);

  /// Apply a power modifier to a collection of gates
  ///
  /// @param exponent The exponent to raise the gates to
  /// @param qubits Input qubits
  /// @param body Function that builds the body containing the gates to
  /// exponentiate
  /// @return Output qubits
  ///
  /// @par Example:
  /// ```c++
  /// qubits_out = builder.pow(2.0, {q0_in, q1_in},
  ///   [&](ValueRange qubits) -> SmallVector<Value> {
  ///     auto [q0, q1] = builder.swap(qubits[0], qubits[1]);
  ///     return {q0, q1};
  ///   }
  /// );
  /// ```
  /// ```mlir
  /// %q_out = qco.pow(%exponent) (%q = %q_in) {
  ///   %q_res = qco.s %q : !qco.qubit -> !qco.qubit
  ///   qco.yield %q_res
  /// } : {!qco.qubit} -> {!qco.qubit}
  /// ```
  ValueRange pow(const std::variant<double, Value>& exponent, ValueRange qubits,
                 function_ref<SmallVector<Value>(ValueRange)> body);

  /// Apply a power modifier on a single qubit.
  ///
  /// @param exponent The exponent to raise the operation to
  /// @param qubit Input qubit
  /// @param body Function that builds the body containing the operation to
  /// exponentiate
  /// @return Output qubit
  ///
  /// @par Example:
  /// ```c++
  /// auto qubit_out = builder.pow(2.0, q0_in, [&](Value qubit) {
  ///   return builder.s(qubit);
  /// });
  /// ```
  Value pow(const std::variant<double, Value>& exponent, Value qubit,
            function_ref<Value(Value)> body);

  //===--------------------------------------------------------------------===//
  // Deallocation
  //===--------------------------------------------------------------------===//

  /// Consume a qubit value (end of lifetime)
  ///
  /// Validates and removes the qubit from tracking. Optional; `finalize()`
  /// automatically sinks all remaining qubits.
  ///
  /// @param qubit Qubit to sink (must be valid/unconsumed)
  /// @return Reference to this builder for method chaining
  ///
  /// @par Example:
  /// ```c++
  /// builder.sink(q);
  /// ```
  /// ```mlir
  /// qco.sink %q : !qco.qubit
  /// ```
  QCOProgramBuilder& sink(Value qubit);

  //===--------------------------------------------------------------------===//
  // SCF operations
  //===--------------------------------------------------------------------===//

  /// Construct an if operation for qubits or tensors of qubits with
  /// linear typing
  ///
  /// Constructs an if operation that takes a bool Value and a range of qubit
  /// and qtensor values that are used in the then/else region of this
  /// operation. The values are passed down as block arguments to each region.
  /// Qubits that were extracted from a tensor that is used as an argument for
  /// this operation are automatically inserted before the operation is
  /// constructed.
  ///
  /// @param condition Bool condition
  /// @param initArgs Initial arguments for the if branches
  /// @param thenBody Function that builds the then body of the if operation
  /// @param elseBody Function that builds the else body of the if operation
  /// @return ValueRange of the results
  ///
  /// @par Example:
  /// ```c++
  /// result = builder.qcoIf(condition, initArgs, [&](ValueRange args)
  /// -> SmallVector<Value> {
  ///   auto q1 = builder.x(args[0]);
  ///   return {q1};
  /// }, [&](ValueRange args) -> SmallVector<Value> {
  ///   auto q2 = builder.z(args[0]);
  ///   return {q2};
  /// });
  /// ```
  /// ```mlir
  /// %q3 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
  ///   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  ///   qco.yield %q1 : !qco.qubit
  /// } else args(%arg0 = %q0) {
  ///   %q2 = qco.z %arg0 : !qco.qubit -> !qco.qubit
  ///   qco.yield %q2 : !qco.qubit
  /// }
  /// ```
  ValueRange
  qcoIf(const std::variant<bool, Value>& condition, ValueRange initArgs,
        function_ref<SmallVector<Value>(ValueRange)> thenBody,
        function_ref<SmallVector<Value>(ValueRange)> elseBody = nullptr);

  /// Construct an scf.if operation conditioned on a classical bit
  ///
  /// Loads the classical bit from the given classical register at the
  /// given index and uses it as the condition of the if operation.
  ///
  /// @param reg The CBit register
  /// @param index The index within the register to load the condition from
  /// @param initArgs Initial arguments threaded through the if operation
  /// @param thenBody Function that builds the then body of the if operation
  /// @param elseBody Function that builds the else body of the if operation
  /// @return ValueRange of the results
  ValueRange
  qcoIf(Value reg, const std::variant<int64_t, Value>& index,
        ValueRange initArgs,
        function_ref<SmallVector<Value>(ValueRange)> thenBody,
        function_ref<SmallVector<Value>(ValueRange)> elseBody = nullptr);

  /// Construct an if operation for qubits with a single target qubit or
  /// tensor.
  ///
  /// Constructs an if operation that takes a bool Value and a single qubit
  /// or qtensor value that is used in the then/else region of this operation.
  /// The value is passed down as block arguments to each region. Qubits that
  /// were extracted from a tensor that is used as an argument for this
  /// operation are automatically inserted before the operation is constructed.
  ///
  /// @param condition Bool condition
  /// @param initArg Initial argument for the if branches
  /// @param thenBody Function that builds the then body of the if operation
  /// @param elseBody Function that builds the else body of the if operation
  /// @return Value as a result
  ///
  /// @par Example:
  /// ```c++
  /// result = builder.qcoIf(condition, initArg, [&](Value arg)
  /// -> Value {
  ///   auto q1 = builder.x(arg);
  ///   return q1;
  /// }, [&](Value arg) -> Value {
  ///   auto q2 = builder.z(arg);
  ///   return q2;
  /// });
  /// ```
  /// ```mlir
  /// %q3 = qco.if %condition args(%arg0 = %q0) -> (!qco.qubit) {
  ///   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  ///   qco.yield %q1 : !qco.qubit
  /// } else args(%arg0 = %q0) {
  ///   %q2 = qco.z %arg0 : !qco.qubit -> !qco.qubit
  ///   qco.yield %q2 : !qco.qubit
  /// }
  /// ```
  Value qcoIf(const std::variant<bool, Value>& condition, Value initArg,
              function_ref<Value(Value)> thenBody,
              function_ref<Value(Value)> elseBody = nullptr);

  /// Construct an index switch operation for qubits or tensors of qubits
  /// with linear typing.
  ///
  /// Constructs an index switch operation that takes an index Value and a range
  /// of qubit and qtensor values that are used in the case regions of this
  /// operation. The values are passed down as block arguments to each region.
  /// Qubits that were extracted from a tensor that is used as an argument for
  /// this operation are automatically inserted before the operation is
  /// constructed.
  ///
  /// @param arg Index argument.
  /// @param targets Initial arguments for the index switch branches.
  /// @param cases The individual switch cases.
  /// @param caseBodies An array of functions that build the case bodies.
  /// @param defaultBody Function that builds the default body.
  /// @return ValueRange of the results.
  ///
  /// @par Example:
  /// ```c++
  /// result = b.qcoIndexSwitch(arg, initTargets,
  ///   SmallVector<int64_t>{0},
  ///   SmallVector<function_ref<SmallVector<Value>(ValueRange)>>{
  ///     [&](ValueRange args) {
  ///       auto q1 = builder.x(args[0]);
  ///       return {q1};
  ///     }
  ///   },
  ///   [&](ValueRange args) {
  ///     auto q2 = builder.x(args[0]);
  ///     return {q2};
  ///   });
  /// ```
  /// ```mlir
  /// %result = qco.index_switch %arg -> !qco.qubit
  /// case 0 args(%arg0 = %q0) {
  ///   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
  ///   qco.yield %q1 : !qco.qubit
  /// }
  /// default args(%arg0 = %q0) {
  ///   %q2 = qco.z %arg0 : !qco.qubit -> !qco.qubit
  ///   qco.yield %q2 : !qco.qubit
  /// }
  /// ```
  ValueRange qcoIndexSwitch(
      const std::variant<int64_t, Value>& arg, ValueRange targets,
      ArrayRef<int64_t> cases,
      ArrayRef<function_ref<SmallVector<Value>(ValueRange)>> caseBodies,
      function_ref<SmallVector<Value>(ValueRange)> defaultBody);

  /// Construct an index switch operation with a single linear target.
  ///
  /// Constructs an index switch operation for one qubit or qtensor value.
  /// Each branch callback receives and returns a single value, avoiding
  /// one-element ranges and vectors.
  ///
  /// @param arg Index argument.
  /// @param target Initial argument for every index switch branch.
  /// @param cases The individual switch cases.
  /// @param caseBodies Functions that build the case bodies.
  /// @param defaultBody Function that builds the default body.
  /// @return The single result value.
  ///
  /// @par Example:
  /// ```c++
  /// result = builder.qcoIndexSwitch(
  ///     arg, target, SmallVector<int64_t>{0},
  ///     SmallVector<function_ref<Value(Value)>>{
  ///         [&](Value value) { return builder.x(value); }},
  ///     [&](Value value) { return builder.z(value); });
  /// ```
  Value qcoIndexSwitch(const std::variant<int64_t, Value>& arg, Value target,
                       ArrayRef<int64_t> cases,
                       ArrayRef<function_ref<Value(Value)>> caseBodies,
                       function_ref<Value(Value)> defaultBody);

  /// Construct an scf.for operation
  ///
  /// Constructs an scf.for operation with the given loop boundaries and
  /// stepsize and a range of qubit and qtensor values for its iter args. Qubits
  /// that were extracted from a tensor that is used as an argument for this
  /// operation are automatically inserted before the operation is constructed.
  ///
  /// @param lowerbound Lower bound of the loop
  /// @param upperbound Upper bound of the loop
  /// @param step Step size of the loop
  /// @param initArgs Initial arguments for the iter args
  /// @param body Function that builds the body of the for operation
  /// @return ValueRange of the results
  ///
  /// @par Example:
  /// ```c++
  /// builder.scfFor(lb, ub, step, initArgs, [&](Value iv, ValueRange iterArgs)
  /// -> SmallVector<Value> {
  ///   auto [t0, q0] = builder.qtensorExtract(iterArgs[0], iv);
  ///   auto q1 = builder.h(q0);
  ///   auto insert = builder.qtensorInsert(q1, t0, iv);
  ///   return {insert};
  /// });
  /// ```
  /// ```mlir
  /// %t3 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %t0)
  /// -> (tensor<3x!qco.qubit>) {
  ///   %t1, %q0 = qtensor.extract %arg0[%iv] : tensor<3x!qco.qubit>
  ///   %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
  ///   %t2 = qtensor.insert %q1 into %t1[%iv] : tensor<3x!qco.qubit>
  ///   scf.yield %t2 : tensor<3x!qco.qubit>
  /// }
  /// ```
  ValueRange scfFor(const std::variant<int64_t, Value>& lowerbound,
                    const std::variant<int64_t, Value>& upperbound,
                    const std::variant<int64_t, Value>& step,
                    ValueRange initArgs,
                    function_ref<SmallVector<Value>(Value, ValueRange)> body);

  /// Construct an scf.while operation
  ///
  /// Constructs an scf.while with a range of qubit and qtensor values for its
  /// iter args. Qubits that were extracted from a tensor that is used as an
  /// argument for this operation are automatically inserted before the
  /// operation is constructed.
  ///
  /// @param initArgs Arguments for the while loop
  /// @param beforeBody Function that builds the before body of the while
  /// operation
  /// @param afterBody Function that builds the after body of the while
  /// operation
  /// @return ValueRange of the results
  ///
  /// @par Example:
  /// ```c++
  /// builder.scfWhile(initArgs, [&](ValueRange iterArgs) ->
  /// SmallVector<Value> {
  ///   auto [q0, cond] = builder.measure(iterArgs[0]);
  ///   builder.scfCondition(cond, q0);
  ///   return {q0};
  /// }, [&](ValueRange iterArgs) -> SmallVector<Value> {
  ///   auto q0 = builder.h(iterArgs[0]);
  ///   return {q0};
  /// });
  /// ```
  /// ```mlir
  /// %q2 = scf.while (%arg0 = %q0): (!qco.qubit) -> (!qco.qubit) {
  ///   %q1, %cond = qco.measure %arg0 : !qco.qubit
  ///   scf.condition(%cond) %q1 : !qco.qubit
  /// } do {
  /// ^bb0(%arg0 : !qco.qubit):
  ///   %q1 = qco.h %arg0 : !qco.qubit -> !qco.qubit
  ///   scf.yield %q1 : !qco.qubit
  /// }
  /// ```
  ValueRange scfWhile(ValueRange initArgs,
                      function_ref<SmallVector<Value>(ValueRange)> beforeBody,
                      function_ref<SmallVector<Value>(ValueRange)> afterBody);

  /// Construct an scf.condition operation with yielded values
  ///
  /// @param condition Condition for the condition operation
  /// @param yieldedValues ValueRange of the yieldedValues
  /// @return Reference to this builder for method chaining
  ///
  /// @par Example:
  /// ```c++
  /// builder.scfCondition(condition, q0);
  /// ```
  /// ```mlir
  /// scf.condition(%condition) %q0 : !qco.qubit
  /// ```
  QCOProgramBuilder& scfCondition(Value condition, ValueRange yieldedValues);

  /// Construct an scf.condition operation conditioned on a classical bit
  ///
  /// Loads the classical bit from the given classical register at the
  /// given index and uses it as the condition of the condition operation.
  ///
  /// @param reg The CBit register
  /// @param index The index within the register to load the condition from
  /// @param yieldedValues ValueRange of the yielded values
  /// @return Reference to this builder for method chaining
  QCOProgramBuilder& scfCondition(Value reg,
                                  const std::variant<int64_t, Value>& index,
                                  ValueRange yieldedValues);

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  /// Finalize the program and return the constructed module
  ///
  /// Automatically deallocates all remaining valid qubits and tensors of
  /// qubits, adds a return statement with exit code 0 (indicating successful
  /// execution), and transfers ownership of the module to the caller. The
  /// builder should not be used after calling this method.
  ///
  /// @return OwningOpRef containing the constructed quantum program module
  OwningOpRef<ModuleOp> finalize();

  /// Finalize the program with the given return values and return the
  /// constructed module
  /// @param returnValues Values representing the return values of the main
  /// function.
  ///
  /// Automatically deallocates all remaining valid qubits and tensors of
  /// qubits, adds a return statement with the given return values, and
  /// transfers ownership of the module to the caller. The builder should not
  /// be used after calling this method.
  ///
  /// The return values must have the types indicated by the function signature
  /// of the main function, which returns an `i64` by default and can be
  /// modified by passing different arguments to the `initialize()` method.
  ///
  /// @return OwningOpRef containing the constructed quantum program module
  OwningOpRef<ModuleOp> finalize(ValueRange returnValues);

  /// Convenience method for building quantum programs.
  /// @param context The MLIR context to use for building the program
  /// @param buildFunc A function that takes a reference to a QCOProgramBuilder
  /// and uses it to build the desired quantum program. The builder will be
  /// properly initialized before calling this function, and the resulting
  /// module will be finalized using the returned Values after this function
  /// completes.
  /// @return The module containing the quantum program built by buildFunc.
  static OwningOpRef<ModuleOp>
  build(MLIRContext* context,
        const function_ref<SmallVector<Value>(QCOProgramBuilder&)>& buildFunc);

  /// Convenience method for building quantum programs with one return
  /// value.
  /// @param context The MLIR context to use for building the program
  /// @param buildFunc A function that takes a reference to a QCOProgramBuilder
  /// and returns the single result value of the desired quantum program.
  /// @return The module containing the quantum program built by buildFunc.
  static OwningOpRef<ModuleOp>
  build(MLIRContext* context,
        const function_ref<Value(QCOProgramBuilder&)>& buildFunc);

private:
  enum class AllocationMode : uint8_t { Unset, Static, Dynamic };

  MLIRContext* ctx{};
  Operation* moduleOp_;

  /// Check if the builder has been finalized
  void checkFinalized() const;

  //===--------------------------------------------------------------------===//
  // Linear Type Tracking Helpers
  //===--------------------------------------------------------------------===//

  /// Validate that a qubit value is valid and unconsumed
  /// @param qubit Qubit value to validate
  /// @throws Aborts if qubit is not tracked (consumed or never created)
  void validateQubitValue(Value qubit) const;

  /// Update tracking when an operation consumes and produces a qubit
  /// @param inputQubit Input qubit being consumed (must be valid)
  /// @param outputQubit New output qubit being produced
  void updateQubitTracking(Value inputQubit, Value outputQubit);

  /// Count unique tensors
  int64_t tensorCounter = 0;

  struct QubitDenseMapInfo {
    static unsigned getHashValue(const Qubit& qubit) {
      return llvm::DenseMapInfo<Value>::getHashValue(qubit.value);
    }
    static bool isEqual(const Qubit& lhs, const Qubit& rhs) {
      return lhs.value == rhs.value;
    }
  };

  /// Track valid (unconsumed) qubit SSA values for linear type enforcement.
  /// Only values present in this set are valid for use in operations.
  /// When an operation consumes a qubit and produces a new one, the old value
  /// is removed and the new output is added.
  DenseSet<Qubit, QubitDenseMapInfo> validQubits;

  /// Validate that a tensor value is valid and unconsumed. This also
  /// checks if the tensor is one-dimensional and contains !qco.qubit as its
  /// values
  /// @param tensor Tensor value to validate
  /// @throws Aborts if tensor is not tracked (consumed or never created)
  void validateTensorValue(Value tensor) const;

  /// Update tracking when an operation consumes and produces a tensor
  /// @param inputTensor Input tensor being consumed (must be valid)
  /// @param outputTensor New output tensor being produced
  void updateTensorTracking(Value inputTensor, Value outputTensor);

  /// Dispose of every live linear value in the current function.
  void disposeLinearValues();

  /// Prepares initial arguments for operations by re-inserting extracted
  /// qubits into their tensors
  ///
  /// For each tensor in @p initArgs, any qubits extracted from it that
  /// are not also present in @p initArgs are inserted back. The latest tensor
  /// values after inserting the qubits are returned. Qubit values are returned
  /// without modifications.
  ///
  /// @param initArgs ValueRange of the initial values
  /// @return SmallVector of the updated values of the initial values.
  SmallVector<Value> prepareInitArgs(ValueRange initArgs);

  /// Prepare one initial argument by re-inserting extracted qubits into
  /// its tensor, if necessary.
  /// @param initArg Initial value
  /// @return Updated initial value
  Value prepareInitArg(Value initArg);

  Value prepareInitArg(Value initArg, const DenseSet<Value>* initQubits);

  /// Update linear-value tracking for one replaced value
  /// @param oldValue The old value to be replaced
  /// @param newValue The new value to be tracked
  void updateQubitValueTracking(Value oldValue, Value newValue);

  /// Update the qubit tracking of the old values with the new values
  /// @param oldValues The old values to be replaced
  /// @param newValues The new values to be tracked
  void updateQubitValueTracking(ValueRange oldValues, ValueRange newValues);

  /// Check if every value is either a qubit or a tensor of qubits
  /// @param values The values that are checked
  /// @throws Abort if a value is neither a qubit nor a tensor of qubits
  static void checkQubitType(ValueRange values);

  struct TensorDenseMapInfo {
    static unsigned getHashValue(const Tensor& tensor) {
      return llvm::DenseMapInfo<Value>::getHashValue(tensor.value);
    }
    static bool isEqual(const Tensor& lhs, const Tensor& rhs) {
      return lhs.value == rhs.value;
    }
  };

  /// Track valid (unconsumed) tensor SSA values for linear type enforcement.
  /// Only values present in this set are valid for use in operations.
  /// When an operation consumes a tensor and produces a new one, the old value
  /// is removed and the new output is added.
  DenseSet<Tensor, TensorDenseMapInfo> validTensors;

  /// Track whether static or dynamic qubit allocation is used.
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Ensure static and dynamic qubit allocation modes are not mixed.
  void ensureAllocationMode(AllocationMode requestedMode);
};
} // namespace qco
} // namespace mlir
