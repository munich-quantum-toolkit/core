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

#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace mlir {
// Forward declarations
class Block;
class MLIRContext;
class ModuleOp;
class Operation;
class ValueRange;

namespace qir {

/**
 * @brief Builder API for constructing QIR (Quantum Intermediate
 * Representation) programs
 *
 * @details
 * The QIRProgramBuilder provides a type-safe interface for constructing
 * quantum programs in QIR format. Like QC, QIR uses reference semantics
 * where operations modify qubits in place, but QIR programs require specific
 * boilerplate structure including proper block organization and metadata
 * attributes.
 *
 * @par QIR Base Profile Structure:
 * QIR Base Profile compliant programs follow a specific 4-block structure:
 * - Entry block: Constants and initialization (__quantum__rt__initialize)
 * - Body block: Reversible quantum operations (gates)
 * - Measurements block: Measurements, resets, deallocations
 * - Output block: Output recording calls (array-based, grouped by register)
 *
 * @par Qubit addressing:
 * A program must use either static qubits (`staticQubit`) or dynamic allocation
 * (`allocQubit`, `allocQubitRegister`), never both. The builder terminates
 * with a usage error if the modes are mixed.
 *
 * @par Example Usage:
 * ```c++
 * QIRProgramBuilder builder(context);
 * builder.initialize();
 *
 * auto q0 = builder.staticQubit(0);
 * auto q1 = builder.staticQubit(1);
 *
 * // Operations use QIR function calls
 * builder.h(q0).cx(q0, q1);
 *
 * // Measure with register info for proper output recording
 * auto c = builder.allocClassicalBitRegister(2, "c");
 * builder.measure(q0, c[0]);
 * builder.measure(q1, c[1]);
 *
 * auto module = builder.finalize();
 * ```
 */
class QIRProgramBuilder final : public ImplicitLocOpBuilder {
public:
  enum class Profile : uint8_t { Base, Adaptive };

  /**
   * @brief Construct a new QIRProgramBuilder
   * @param context The MLIR context to use for building operations
   */
  explicit QIRProgramBuilder(MLIRContext* context);

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Initialize the builder and prepare for program construction
   *
   * @details
   * Creates the main function with proper QIR structure (4-block layout),
   * adds __quantum__rt__initialize call, and sets up the builder's insertion
   * points. Must be called before adding operations.
   */
  void initialize();

  /**
   * @brief Initialize the builder and prepare for program construction
   * with specified return types.
   * @param returnType The return type for the main function
   *
   * @details
   * Creates a main function with an entry_point attribute. Must be called
   * before adding operations.
   */
  void initialize(Type returnType);

  /**
   * @brief Modify the return type of the main function after initialization.
   * @param returnType The new return type for the main function
   */
  void retype(Type returnType);

  //===--------------------------------------------------------------------===//
  // Constants
  //===--------------------------------------------------------------------===//

  /**
   * @brief Create a constant integer value
   * @param value The value to store in the constant
   * @return The value produced by the constant operation
   *
   * @par Example:
   * ```c++
   * auto c = builder.intConstant(1);
   * ```
   * ```mlir
   * %c = arith.constant 1 : i64
   * ```
   */
  Value intConstant(int64_t value);

  /**
   * @brief Create a constant double value
   * @param value The value to store in the constant
   * @return The value produced by the constant operation
   *
   * @par Example:
   * ```c++
   * auto c = builder.doubleConstant(0.5);
   * ```
   * ```mlir
   * %c = arith.constant 0.5 : f64
   * ```
   */
  Value doubleConstant(double value);

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /**
   * @brief Allocate a qubit
   * @return An LLVM pointer representing the qubit
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubit();
   * ```
   * ```mlir
   * %zero = llvm.mlir.zero : !llvm.ptr
   * %q = llvm.call @"@__quantum__rt__qubit_allocate"(%zero) : !llvm.ptr ->
   * !llvm.ptr
   * ```
   */
  Value allocQubit();

  /**
   * @brief Get a static qubit by index
   * @param index The qubit index (must be non-negative)
   * @return An LLVM pointer representing the qubit
   *
   * @par Example:
   * ```c++
   * auto q0 = builder.staticQubit(0);
   * ```
   * ```mlir
   * %c0 = llvm.mlir.constant(0 : i64) : i64
   * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
   * ```
   */
  Value staticQubit(int64_t index);

  /**
   * @brief Get a static result by index
   * @param index The result index (must be non-negative)
   * @return An LLVM pointer representing the result
   *
   * @par Example:
   *
   */
  Value staticResult(int64_t index);

  /**
   * @brief Represents a qubit register with its qubits.
   */
  struct QubitRegister {
    /// The llvm.ptr value representing the qubit register
    Value value;
    /// The allocated qubit values
    SmallVector<Value> qubits;

    /**
     * @brief Access a specific qubit in the register
     * @param index The index of the qubit to access
     * @return The specified qubit value
     */
    Value operator[](size_t index) const;

    /**
     * @brief Conversion to the backing MemRef value
     * @return The llvm.ptr value representing the qubit register
     */
    explicit operator Value() const { return value; }
  };

  /**
   * @brief Allocate an array of qubits
   * @param size Number of qubits (must be positive)
   * @return A `QubitRegister` structure
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubitRegister(3);
   * ```
   * ```mlir
   * %zero = llvm.mlir.zero : !llvm.ptr
   * %alloca = llvm.alloca %c3 x !llvm.ptr : (i64) -> !llvm.ptr
   * llvm.call @"@__quantum__rt__qubit_array_allocate"(%c3, %alloca, %zero) :
   * (i64, !llvm.ptr, !llvm.ptr) -> ()
   * %q0 = llvm.load %alloca : !llvm.ptr -> !llvm.ptr
   * %ptr1 = llvm.getelementptr %alloca[1] : !llvm.ptr -> !llvm.ptr
   * %q1 = llvm.load %ptr1 : !llvm.ptr -> !llvm.ptr
   * %ptr2 = llvm.getelementptr %alloca[2] : !llvm.ptr -> !llvm.ptr
   * %q2 = llvm.load %ptr2 : !llvm.ptr -> !llvm.ptr
   * ```
   */
  QubitRegister allocQubitRegister(int64_t size);

  /**
   * @brief Loads a qubit from a register
   *
   * @param reg Source register
   * @param index The index from where the qubit is loaded
   * @return The loaded qubit
   *
   * @par Example:
   * ```c++
   * auto q0 = builder.load(register, index);
   * ```
   * ```mlir
   * %gep = llvm.getelementptr %alloc[%index] : (!llvm.ptr, i64) -> !llvm.ptr,
   * !llvm.ptr
   * %q0 = llvm.load %gep : !llvm.ptr -> !llvm.ptr
   * ```
   */
  Value load(Value reg, Value index);

  /**
   * @brief A small structure representing a single classical bit within a
   * classical register.
   */
  struct Bit {
    /// Name of the register containing this bit
    StringRef registerName;
    /// Size of the register containing this bit
    int64_t registerSize{};
    /// Index of this bit within the register
    int64_t registerIndex{};
  };

  /**
   * @brief A small structure representing a classical bit register.
   */
  struct ClassicalRegister {
    /// Name of the classical register
    std::string name;
    /// Size of the classical register
    int64_t size;

    /**
     * @brief Access a specific bit in the classical register
     * @param index The index of the bit to access (must be less than size)
     * @return A Bit structure representing the specified bit
     */
    Bit operator[](const int64_t index) const;
  };

  /**
   * @brief Allocate a classical bit register
   * @param size Number of bits
   * @param name Register name (default: "c")
   * @return A ClassicalRegister structure
   *
   * @par Example:
   * ```c++
   * auto c = builder.allocClassicalBitRegister(3, "c");
   * ```
   */
  ClassicalRegister allocClassicalBitRegister(int64_t size,
                                              const std::string& name = "c");

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measure a qubit and record the result (simple version)
   *
   * @details
   * Performs a Z-basis measurement using `__quantum__qis__mz__body`.
   *
   * The output is recorded via `__quantum__rt__result_record_output` during
   * `finalize()`.
   *
   * @param qubit The qubit to measure
   * @param resultIndex The classical bit index for result pointer
   * @param record Whether the measurement should be recorded in the output
   * @return An LLVM pointer to the measurement result
   *
   * @par Example:
   * ```c++
   * auto result = builder.measure(q0, 0);
   * ```
   * ```mlir
   * // In entry block:
   * %zero = llvm.mlir.zero : !llvm.ptr
   * %r = llvm.call @"@__quantum__rt__result_allocate"(%zero) : !llvm.ptr ->
   * !llvm.ptr
   *
   * // In measurements block:
   * llvm.call @__quantum__qis__mz__body(%q0, %r) : (!llvm.ptr, !llvm.ptr) -> ()
   *
   * // In output block:
   * llvm.call @__quantum__rt__result_record_output(%r, %label) : (!llvm.ptr,
   * !llvm.ptr) -> ()
   * ```
   */
  Value measure(Value qubit, int64_t resultIndex, bool record = true);

  /**
   * @brief Measure a qubit into a classical register
   *
   * @details
   * Performs a Z-basis measurement using `__quantum__qis__mz__body`.
   *
   * The output is recorded via `__quantum__rt__result_array_record_output`
   * during `finalize()`.
   *
   * @param qubit The qubit to measure
   * @param bit The classical bit to store the result
   * @param record Whether the measurement should be recorded in the output
   * @return An LLVM pointer to the measurement result
   *
   * @par Example:
   * ```c++
   * auto c = builder.allocClassicalBitRegister(2, "c");
   * builder.measure(q0, c[0]);
   * ```
   * ```mlir
   * // In entry block:
   * %zero = llvm.mlir.zero : !llvm.ptr
   * %alloca = llvm.alloca %c2 x !llvm.ptr : (i64) -> !llvm.ptr
   * llvm.call @"@__quantum__rt__result_array_allocate"(%c2, %alloca, %zero) :
   * (i64, !llvm.ptr, !llvm.ptr) -> ()
   * %r = llvm.load %alloca : !llvm.ptr -> !llvm.ptr
   *
   * // In measurements block:
   * llvm.call @__quantum__qis__mz__body(%q, %r) : (!llvm.ptr, !llvm.ptr) -> ()
   *
   * // In output block:
   * llvm.call @__quantum__rt__result_array_record_output(%c2, %alloca, %label)
   * : (i64, !llvm.ptr, !llvm.ptr) -> ()
   * ```
   */
  Value measure(Value qubit, const Bit& bit, bool record = true);

  /**
   * @brief Reset a qubit to |0⟩ state
   *
   * @details
   * Resets a qubit using __quantum__qis__reset__body.
   *
   * @param qubit The qubit to reset
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.reset(q);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__reset__body(%q) : !llvm.ptr -> ()
   * ```
   */
  QIRProgramBuilder& reset(Value qubit);

  //===--------------------------------------------------------------------===//
  // Unitary Operations
  //===--------------------------------------------------------------------===//

  // GPhaseOp

  /**
   * @brief Apply a QIR gphase operation
   *
   * @param theta Rotation angle in radians
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.gphase(theta);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__gphase__body(%theta) : (f64) -> ()
   * ```
   */
  QIRProgramBuilder& gphase(const std::variant<double, Value>& theta);

  // OneTargetZeroParameter

#define DECLARE_ONE_TARGET_ZERO_PARAMETER(OP_NAME, QIR_NAME)                   \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(q);                                                       \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q) : !llvm.ptr -> ()       \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(Value qubit);                                     \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param control Control qubit                                              \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(q0, q1);                                               \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1) : (!llvm.ptr,    \
   * !llvm.ptr) -> ()                                                          \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(Value control, Value target);                  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param controls Control qubits                                            \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME({q0, q1}, q2);                                        \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2) :          \
   * (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()                                   \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(ValueRange controls, Value target);

  DECLARE_ONE_TARGET_ZERO_PARAMETER(id, i)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(x, x)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(y, y)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(z, z)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(h, h)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(s, s)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(t, t)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(sx, sx)

#undef DECLARE_ONE_TARGET_ZERO_PARAMETER

#define DECLARE_ONE_TARGET_ZERO_PARAMETER_ADJOINT(OP_NAME, QIR_NAME)           \
  /**                                                                          \
   * @brief Apply an adjoint QIR_NAME operation                                \
   *                                                                           \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(q);                                                       \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__adj(%q) : !llvm.ptr -> ()        \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(Value qubit);                                     \
  /**                                                                          \
   * @brief Apply an adjoint controlled QIR_NAME operation                     \
   *                                                                           \
   * @param control Control qubit                                              \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(q0, q1);                                               \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__adj(%q0, %q1) : (!llvm.ptr,     \
   * !llvm.ptr) -> ()                                                          \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(Value control, Value target);                  \
  /**                                                                          \
   * @brief Apply an adjoint multi-controlled QIR_NAME operation               \
   *                                                                           \
   * @param controls Control qubits                                            \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME({q0, q1}, q2);                                        \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__adj(%q0, %q1, %q2) :           \
   * (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()                                   \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(ValueRange controls, Value target);

  DECLARE_ONE_TARGET_ZERO_PARAMETER_ADJOINT(sdg, s)
  DECLARE_ONE_TARGET_ZERO_PARAMETER_ADJOINT(tdg, t)
  DECLARE_ONE_TARGET_ZERO_PARAMETER_ADJOINT(sxdg, sx)

#undef DECLARE_ONE_TARGET_ZERO_PARAMETER_ADJOINT

  // OneTargetOneParameter

#define DECLARE_ONE_TARGET_ONE_PARAMETER(OP_NAME, QIR_NAME, PARAM)             \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM, q);                                                \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q, %PARAM) : (!llvm.ptr,   \
   * f64) -> ()                                                                \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM),        \
                             Value qubit);                                     \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Control qubit                                              \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM, q0, q1);                                        \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1, %PARAM) :        \
   * (!llvm.ptr, !llvm.ptr, f64) -> ()                                         \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM),     \
                                Value control, Value target);                  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Control qubits                                            \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM, {q0, q1}, q2);                                 \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2, %PARAM) :  \
   * (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64) -> ()                              \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM),    \
                                 ValueRange controls, Value target);

  DECLARE_ONE_TARGET_ONE_PARAMETER(rx, rx, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(ry, ry, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(rz, rz, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(p, p, theta)

#undef DECLARE_ONE_TARGET_ONE_PARAMETER

  // OneTargetTwoParameter

#define DECLARE_ONE_TARGET_TWO_PARAMETER(OP_NAME, QIR_NAME, PARAM1, PARAM2)    \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM1, PARAM2, q);                                       \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q, %PARAM1, %PARAM2) :     \
   * (!llvm.ptr, f64, f64) -> ()                                               \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM1),       \
                             const std::variant<double, Value>&(PARAM2),       \
                             Value qubit);                                     \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param control Control qubit                                              \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM1, PARAM2, q0, q1);                               \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1, %PARAM1,         \
   * %PARAM2) : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()                         \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM1),    \
                                const std::variant<double, Value>&(PARAM2),    \
                                Value control, Value target);                  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param controls Control qubits                                            \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM1, PARAM2, {q0, q1}, q2);                        \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2, %PARAM1,   \
   * %PARAM2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64) -> ()              \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM1),   \
                                 const std::variant<double, Value>&(PARAM2),   \
                                 ValueRange controls, Value target);

  DECLARE_ONE_TARGET_TWO_PARAMETER(r, r, theta, phi)
  DECLARE_ONE_TARGET_TWO_PARAMETER(u2, u2, phi, lambda)

#undef DECLARE_ONE_TARGET_TWO_PARAMETER

  // OneTargetThreeParameter

#define DECLARE_ONE_TARGET_THREE_PARAMETER(OP_NAME, QIR_NAME, PARAM1, PARAM2,  \
                                           PARAM3)                             \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param PARAM3 Rotation angle in radians                                   \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM1, PARAM2, PARAM3, q);                               \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q, %PARAM1, %PARAM2,       \
   * %PARAM3) :                                                                \
   * (!llvm.ptr, f64, f64, f64) -> ()                                          \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM1),       \
                             const std::variant<double, Value>&(PARAM2),       \
                             const std::variant<double, Value>&(PARAM3),       \
                             Value qubit);                                     \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param PARAM3 Rotation angle in radians                                   \
   * @param control Control qubit                                              \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM1, PARAM2, PARAM3, q0, q1);                       \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1, %PARAM1,         \
   * %PARAM2, %PARAM3) : (!llvm.ptr, !llvm.ptr, f64, f64, f64) -> ()           \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM1),    \
                                const std::variant<double, Value>&(PARAM2),    \
                                const std::variant<double, Value>&(PARAM3),    \
                                Value control, Value target);                  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param PARAM3 Rotation angle in radians                                   \
   * @param controls Control qubits                                            \
   * @param target Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM1, PARAM2, PARAM3, {q0, q1}, q2);                \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2, %PARAM1,   \
   * %PARAM2, %PARAM3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, f64) ->   \
   * ()                                                                        \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM1),   \
                                 const std::variant<double, Value>&(PARAM2),   \
                                 const std::variant<double, Value>&(PARAM3),   \
                                 ValueRange controls, Value target);

  DECLARE_ONE_TARGET_THREE_PARAMETER(u, u3, theta, phi, lambda)

#undef DECLARE_ONE_TARGET_THREE_PARAMETER

  // TwoTargetZeroParameter

#define DECLARE_TWO_TARGET_ZERO_PARAMETER(OP_NAME, QIR_NAME)                   \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(q0, q1);                                                  \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q0, %q1) : (!llvm.ptr,     \
   * !llvm.ptr) -> ()                                                          \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(Value qubit0, Value qubit1);                      \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param control Control qubit                                              \
   * @param target0 Target qubit                                               \
   * @param target1 Target qubit                                               \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(q0, q1);                                               \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1) : (!llvm.ptr,    \
   * !llvm.ptr) -> ()                                                          \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(Value control, Value target0, Value target1);  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param controls Control qubits                                            \
   * @param target0 Target qubit                                               \
   * @param target1 Target qubit                                               \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME({q0, q1}, q2);                                        \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2) :          \
   * (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()                                   \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(ValueRange controls, Value target0,           \
                                 Value target1);

  DECLARE_TWO_TARGET_ZERO_PARAMETER(swap, swap)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(iswap, iswap)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(dcx, dcx)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(ecr, ecr)

#undef DECLARE_TWO_TARGET_ZERO_PARAMETER

  // TwoTargetOneParameter

#define DECLARE_TWO_TARGET_ONE_PARAMETER(OP_NAME, QIR_NAME, PARAM)             \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM, q0, q1);                                           \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q0, %q1, %PARAM) :         \
   * (!llvm.ptr, !llvm.ptr, f64) -> ()                                         \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM),        \
                             Value qubit0, Value qubit1);                      \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Control qubit                                              \
   * @param target0 Target qubit                                               \
   * @param target1 Target qubit                                               \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM, q0, q1, q2);                                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1, %q2, %PARAM) :   \
   * (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64) -> ()                              \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM),     \
                                Value control, Value target0, Value target1);  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Control qubits                                            \
   * @param target0 Target qubit                                               \
   * @param target1 Target qubit                                               \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM, {q0, q1}, q2, q3);                             \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2, %q3,       \
   * %PARAM) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64) -> ()         \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM),    \
                                 ValueRange controls, Value target0,           \
                                 Value target1);

  DECLARE_TWO_TARGET_ONE_PARAMETER(rxx, rxx, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(ryy, ryy, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(rzx, rzx, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(rzz, rzz, theta)

#undef DECLARE_TWO_TARGET_ONE_PARAMETER

  // TwoTargetTwoParameter

#define DECLARE_TWO_TARGET_TWO_PARAMETER(OP_NAME, QIR_NAME, PARAM1, PARAM2)    \
  /**                                                                          \
   * @brief Apply a QIR_NAME operation                                         \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM1, PARAM2, q0, q1);                                  \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q0, %q1, %PARAM1, %PARAM2) \
   * : (!llvm.ptr, !llvm.ptr, f64, f64) -> ()                                  \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM1),       \
                             const std::variant<double, Value>&(PARAM2),       \
                             Value qubit0, Value qubit1);                      \
  /**                                                                          \
   * @brief Apply a controlled QIR_NAME operation                              \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param control Control qubit                                              \
   * @param target0 Target qubit                                               \
   * @param target1 Target qubit                                               \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM1, PARAM2, q0, q1, q2);                           \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__c##QIR_NAME##__body(%q0, %q1, %q2, %PARAM1,    \
   * %PARAM2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64) -> ()              \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM1),    \
                                const std::variant<double, Value>&(PARAM2),    \
                                Value control, Value target0, Value target1);  \
  /**                                                                          \
   * @brief Apply a multi-controlled QIR_NAME operation                        \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param controls Control qubits                                            \
   * @param target0 Target qubit                                               \
   * @param target1 Target qubit                                               \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM1, PARAM2, {q0, q1}, q2, q3);                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__cc##QIR_NAME##__body(%q0, %q1, %q2, %q3,       \
   * %PARAM1, %PARAM2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64,     \
   * f64) -> ()                                                                \
   * ```                                                                       \
   */                                                                          \
  QIRProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM1),   \
                                 const std::variant<double, Value>&(PARAM2),   \
                                 ValueRange controls, Value target0,           \
                                 Value target1);

  DECLARE_TWO_TARGET_TWO_PARAMETER(xx_plus_yy, xx_plus_yy, theta, beta)
  DECLARE_TWO_TARGET_TWO_PARAMETER(xx_minus_yy, xx_minus_yy, theta, beta)

#undef DECLARE_TWO_TARGET_TWO_PARAMETER

  //===--------------------------------------------------------------------===//
  // SCF Operations
  //===--------------------------------------------------------------------===//

  /**
   * @brief Construct a for construct in LLVM dialect
   *
   * @param lowerbound Lower bound of the loop
   * @param upperbound Upper bound of the loop
   * @param step Step size of the loop
   * @param body Function that builds the body of the for operation
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.scfFor(lb, ub, step, [&](Value iv) {
   *   auto q0 = builder.load(register, iv);
   *   builder.h(q0);
   * });
   * ```
   * ```mlir
   *   llvm.br ^condition(%lowerbound : i64)
   * ^condition(%iv: i64):
   *   %condition = llvm.icmp "slt" %iv, %upperbound : i64
   *   llvm.cond_br %condition, ^loop, ^next
   * ^loop:
   *   %gep = llvm.getelementptr %alloc[%iv] : (!llvm.ptr, i64) -> !llvm.ptr,
   *   !llvm.ptr
   *   %q0 = llvm.load %gep : !llvm.ptr -> !llvm.ptr
   *   llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
   *   %nextIv = llvm.add %iv, %step : i64
   *   llvm.br ^condition(%nextIv : i64)
   * ^next:
   * ```
   */
  QIRProgramBuilder& scfFor(const std::variant<int64_t, Value>& lowerbound,
                            const std::variant<int64_t, Value>& upperbound,
                            const std::variant<int64_t, Value>& step,
                            const function_ref<void(Value)>& body);

  /**
   * @brief Construct an if construct in LLVM dialect
   *
   * @param condition Condition for the if operation
   * @param thenBody Function that builds the then body of the if construct
   * @param elseBody Function that builds the else body of the if construct
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.scfIf(condition, [&] {
   *   builder.x(q0);
   * }, [&] {
   *   builder.z(q0);
   * });
   * ```
   * ```mlir
   *   %condition = llvm.call @__quantum__rt__read_result(%result) : (!llvm.ptr)
   *   -> i1
   *   llvm.cond_br %condition, ^then, ^else
   * ^then:
   *   llvm.call @__quantum__qis__x__body(%q0) : (!llvm.ptr) -> ()
   *   llvm.br ^next
   * ^else:
   *   llvm.call @__quantum__qis__z__body(%q0) : (!llvm.ptr) -> ()
   *   llvm.br ^next
   * ^next:
   * ```
   */
  QIRProgramBuilder& scfIf(const std::variant<bool, Value>& condition,
                           const function_ref<void()>& thenBody,
                           const function_ref<void()>& elseBody = nullptr);

  /**
   * @brief Construct a while construct in LLVM dialect
   *
   * @param beforeBody Function that builds the before body of the while
   * construct
   * @param afterBody Function that builds the after body of the while construct
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.scfWhile([&] {
   *   auto res = builder.measure(q0);
   *   return res;
   * }, [&] {
   *   builder.h(q0);
   * });
   * ```
   * ```mlir
   *   llvm.br ^before
   * ^before:
   *   llvm.call @__quantum__qis__mz__body(%q0, %result) : (!llvm.ptr,
   * !llvm.ptr)
   *   -> ()
   *   %condition = llvm.call @__quantum__rt__read_result(%result) : (!llvm.ptr)
   *   -> i1
   *   llvm.cond_br %condition, ^after, ^next
   * ^after:
   *   llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
   *   llvm.br ^bb2
   * ^next:
   * ```
   */
  QIRProgramBuilder& scfWhile(const function_ref<Value()>& beforeBody,
                              const function_ref<void()>& afterBody = nullptr);

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Finalize the program and return the constructed module
   *
   * @details
   * Automatically deallocates all remaining allocated qubits and result
   * pointers, generates output recording in the output block, ensures proper
   * QIR metadata attributes are set, and transfers ownership of the module to
   * the caller. The builder should not be used after calling this method.
   *
   * @return OwningOpRef containing the constructed QIR program module
   */
  OwningOpRef<ModuleOp> finalize();

  /**
   * @brief Finalize the program with a given exit code and return the
   * constructed module
   * @param returnValue The value representing the exit code to return
   *
   * @details
   * Automatically deallocates all remaining valid qubits and tensors of qubits,
   * adds a return statement with a given exit code,
   * and transfers ownership of the module to the caller. The builder should not
   * be used after calling this method.
   *
   * The return value must have the type indicated by the function signature
   * of the main function, which returns an `i64` by default and can be
   * modified by passing different arguments to the `initialize()` method.
   *
   * @return OwningOpRef containing the constructed quantum program module
   */
  OwningOpRef<ModuleOp> finalize(Value returnValue);

  /**
   * @brief Convenience method for building quantum programs
   * @param context The MLIR context to use for building the program
   * @param buildFunc A function that takes a reference to a QIRProgramBuilder
   * and uses it to build the desired quantum program. The builder will be
   * properly initialized before calling this function, and the resulting module
   * will be finalized and returned after this function completes.
   * @return The module containing the quantum program built by buildFunc.
   */
  static OwningOpRef<ModuleOp>
  build(MLIRContext* context,
        const function_ref<
            std::pair<mlir::Value, mlir::Type>(QIRProgramBuilder&)>& buildFunc,
        Profile profile = Profile::Adaptive);

private:
  enum class AllocationMode : uint8_t { Unset, Static, Dynamic };

  /// The main module
  Operation* module{};

  /// The main function
  Operation* mainFunc{};

  /// Allocator and StringSaver for stable StringRefs
  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver stringSaver{allocator};

  /// Entry block: constants and initialization
  Block* entryBlock{};
  /// Body block: reversible operations (gates)
  Block* bodyBlock{};
  /// Measurements block: measurements, resets, deallocations
  Block* measurementsBlock{};
  /// Output block: output recording calls
  Block* outputBlock{};

  /// Exit code constant (created in entry block, used in output block)
  Value exitCode;

  /// Cache static qubit pointers for reuse
  DenseMap<int64_t, Value> staticQubits;

  /// Set of qubit pointers
  DenseSet<Value> qubits;

  /// Set of qubit-array pointers
  DenseSet<Value> qubitArrays;

  /// Map from register name to result-array pointer
  llvm::StringMap<Value> resultArrays;

  /// Map from (register name, index) to loaded result
  DenseMap<std::pair<StringRef, int64_t>, Value> loadedResults;

  /// Map from result index to result pointer for non-register results
  DenseMap<int64_t, Value> resultPtrs;

  /// Set of array register names that should be recorded in the output.
  DenseSet<StringRef> recordedArrays;

  /// Set of unnamed result indices that should be recorded in the output.
  DenseSet<int64_t> recordedIndices;

  /// Map from register to their loaded indices
  DenseMap<Value, DenseSet<Value>> loadedQubits;

  /// Helper variable for storing the LLVM pointer type
  Type ptrType;

  /// Helper variable for storing the LLVM void type
  Type voidType;

  /// The number of used qubits.
  size_t numQubits{0};

  /// The number of result values.
  size_t numResults{0};

  /**
   * @brief Helper to create a LLVM CallOp
   *
   * @param parameters Operation parameters
   * @param controls Control qubits
   * @param targets Target qubits
   * @param fnName Name of the QIR function to call
   */
  void createCallOp(const SmallVector<std::variant<double, Value>>& parameters,
                    ValueRange controls, const SmallVector<Value>& targets,
                    StringRef fnName);

  /**
   * @brief Generate array-based output recording in the output block
   *
   * @details
   * Called by `finalize()` to generate output recording calls for all tracked
   * measurements.
   */
  void generateOutputRecording();

  bool isFinalized = false;

  /// Track whether static or dynamic qubit allocation is used.
  AllocationMode allocationMode = AllocationMode::Unset;

  /// Track whether Base or Adaptive Profile is used.
  Profile profile = Profile::Adaptive;

  /// Check if the builder has been finalized
  void checkFinalized() const;

  /// Ensure static and dynamic qubit allocation modes are not mixed.
  void ensureAllocationMode(AllocationMode requestedMode);

  /**
   * @brief Helper to resolve a variant of either int64_t type or Value Type to
   * a Value
   *
   * @details Helper function to resolve a given variant to a Value. Creates a
   * LLVM ConstantOp from the int value. The created LLVM Constant is of type
   * I64 and has an IndexAttr as its value. If the variant holds a Value, return
   * it directly.
   *
   * @param variant The variant to resolve
   * @return The resolved Value
   */
  Value resolveIntVariant(const std::variant<int64_t, Value>& variant);
};

} // namespace qir
} // namespace mlir
