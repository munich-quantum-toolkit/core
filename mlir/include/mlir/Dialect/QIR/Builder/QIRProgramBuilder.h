/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>
#include <variant>

namespace mlir::qir {

/**
 * @brief Builder API for constructing QIR (Quantum Intermediate
 * Representation) programs
 *
 * @details
 * The QIRProgramBuilder provides a type-safe interface for constructing
 * quantum programs in QIR format. Like Quartz, QIR uses reference semantics
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
 * builder.measure(q0, "c", 0);
 * builder.measure(q1, "c", 1);
 *
 * auto module = builder.finalize();
 * ```
 */
class QIRProgramBuilder {
public:
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

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

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
   * @brief Allocate an array of (static) qubits
   * @param size Number of qubits (must be positive)
   * @return Vector of LLVM pointers representing the qubits
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubitRegister(3);
   * ```
   * ```mlir
   * %c0 = llvm.mlir.constant(0 : i64) : i64
   * %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
   * %c1 = llvm.mlir.constant(1 : i64) : i64
   * %q1 = llvm.inttoptr %c1 : i64 to !llvm.ptr
   * %c2 = llvm.mlir.constant(2 : i64) : i64
   * %q2 = llvm.inttoptr %c2 : i64 to !llvm.ptr
   * ```
   */
  SmallVector<Value> allocQubitRegister(int64_t size);

  /**
   * @brief A small structure representing a single classical bit within a
   * classical register.
   */
  struct Bit {
    /// Name of the register containing this bit
    std::string registerName;
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
    Bit operator[](const int64_t index) const {
      if (index < 0 || index >= size) {
        llvm::reportFatalUsageError("Bit index out of bounds");
      }
      return {
          .registerName = name, .registerSize = size, .registerIndex = index};
    }
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
                                              std::string name = "c");

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measure a qubit and record the result (simple version)
   *
   * @details
   * Performs a Z-basis measurement using __quantum__qis__mz__body. The
   * result is tracked for deferred output recording in the output block.
   * This version does NOT include register information, so output will
   * not be grouped by register.
   *
   * @param qubit The qubit to measure
   * @param resultIndex The classical bit index for result pointer
   * @return An LLVM pointer to the measurement result
   *
   * @par Example:
   * ```c++
   * auto result = builder.measure(q0, 0);
   * ```
   * ```mlir
   * // In measurements block:
   * %c0 = llvm.mlir.constant(0 : i64) : i64
   * %r = llvm.inttoptr %c0 : i64 to !llvm.ptr
   * llvm.call @__quantum__qis__mz__body(%q0, %r) : (!llvm.ptr, !llvm.ptr) -> ()
   *
   * // Output recording deferred to output block
   * ```
   */
  Value measure(Value qubit, int64_t resultIndex);

  /**
   * @brief Measure a qubit into a classical register
   *
   * @details
   * Performs a Z-basis measurement using __quantum__qis__mz__body and tracks
   * the measurement with register information for array-based output recording.
   * Output recording is deferred to the output block during finalize(), where
   * measurements are grouped by register and recorded using:
   * 1. __quantum__rt__array_record_output for each register
   * 2. __quantum__rt__result_record_output for each measurement in the register
   *
   * @param qubit The qubit to measure
   * @param bit The classical bit to store the result
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * auto& c = builder.allocClassicalBitRegister(2, "c");
   * builder.measure(q0, c[0]);
   * builder.measure(q1, c[1]);
   * ```
   * ```mlir
   * // In measurements block:
   * llvm.call @__quantum__qis__mz__body(%q0, %r0) : (!llvm.ptr, !llvm.ptr) ->
   * () llvm.call @__quantum__qis__mz__body(%q1, %r1) : (!llvm.ptr, !llvm.ptr)
   * -> ()
   *
   * // In output block (generated during finalize):
   * @0 = internal constant [3 x i8] c"c\00"
   * @1 = internal constant [5 x i8] c"c0r\00"
   * @2 = internal constant [5 x i8] c"c1r\00"
   * llvm.call @__quantum__rt__array_record_output(i64 2, ptr @0)
   * llvm.call @__quantum__rt__result_record_output(ptr %r0, ptr @1)
   * llvm.call @__quantum__rt__result_record_output(ptr %r1, ptr @2)
   * ```
   */
  QIRProgramBuilder& measure(Value qubit, const Bit& bit);

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
   * llvm.call @__quantum__qis__reset__body(%q) : (!llvm.ptr) -> ()
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
   * @brief Apply a QIR QIR_NAME operation                                     \
   *                                                                           \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(q);                                                       \
   * ```                                                                       \
   * ```mlir                                                                   \
   * llvm.call @__quantum__qis__##QIR_NAME##__body(%q) : (!llvm.ptr) -> ()     \
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
  DECLARE_ONE_TARGET_ZERO_PARAMETER(sdg, sdg)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(t, t)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(tdg, tdg)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(sx, sx)
  DECLARE_ONE_TARGET_ZERO_PARAMETER(sxdg, sxdg)

#undef DECLARE_ONE_TARGET_ZERO_PARAMETER

  // OneTargetOneParameter

#define DECLARE_ONE_TARGET_ONE_PARAMETER(OP_NAME, QIR_NAME, PARAM)             \
  /**                                                                          \
   * @brief Apply a QIR QIR_NAME operation                                     \
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
   * @brief Apply a QIR QIR_NAME operation                                     \
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
   * @brief Apply a QIR QIR_NAME operation                                     \
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
   * @brief Apply a QIR QIR_NAME operation                                     \
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
   * @brief Apply a QIR QIR_NAME operation                                     \
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
   * @brief Apply a QIR QIR_NAME operation                                     \
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
  // Finalization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Finalize the program and return the constructed module
   *
   * @details
   * Automatically deallocates all remaining allocated qubits, generates
   * array-based output recording in the output block (grouped by register),
   * ensures proper QIR metadata attributes are set, and transfers ownership
   * of the module to the caller. The builder should not be used after calling
   * this method.
   *
   * @return OwningOpRef containing the constructed QIR program module
   */
  OwningOpRef<ModuleOp> finalize();

private:
  OpBuilder builder;
  ModuleOp module;
  Location loc;

  LLVM::LLVMFuncOp mainFunc;

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
  LLVM::ConstantOp exitCode;

  /// Cache static pointers for reuse
  DenseMap<int64_t, Value> ptrCache;

  /// Map from (register_name, register_index) to result pointer
  DenseMap<std::pair<StringRef, int64_t>, Value> registerResultMap;

  /// Track qubit and result counts for QIR metadata
  QIRMetadata metadata_;

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
   * Called by finalize() to generate output recording calls for all tracked
   * measurements. Groups measurements by register and generates:
   * 1. array_record_output for each register
   * 2. result_record_output for each measurement in the register
   */
  void generateOutputRecording();

  bool isFinalized = false;

  /// Check if the builder has been finalized
  void checkFinalized() const;
};

} // namespace mlir::qir
