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

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstdint>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <variant>

namespace mlir::quartz {

/**
 * @brief Builder API for constructing quantum programs in the Quartz dialect
 *
 * @details
 * The QuartzProgramBuilder provides a type-safe interface for constructing
 * quantum circuits using reference semantics. Operations modify qubits in
 * place without producing new SSA values, providing a natural mapping to
 * hardware execution models.
 *
 * @par Example Usage:
 * ```c++
 * QuartzProgramBuilder builder(context);
 * builder.initialize();
 *
 * auto q0 = builder.staticQubit(0);
 * auto q1 = builder.staticQubit(1);
 *
 * // Operations modify qubits in place
 * builder.h(q0).cx(q0, q1);
 *
 * auto module = builder.finalize();
 * ```
 */
class QuartzProgramBuilder final : public OpBuilder {
public:
  /**
   * @brief Construct a new QuartzProgramBuilder
   * @param context The MLIR context to use for building operations
   */
  explicit QuartzProgramBuilder(MLIRContext* context);

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Initialize the builder and prepare for program construction
   *
   * @details
   * Creates a main function with an entry_point attribute. Must be called
   * before adding operations.
   */
  void initialize();

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /**
   * @brief Allocate a single qubit initialized to |0⟩
   * @return A qubit reference
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubit();
   * ```
   * ```mlir
   * %q = quartz.alloc : !quartz.qubit
   * ```
   */
  Value allocQubit();

  /**
   * @brief Get a static qubit by index
   * @param index The qubit index (must be non-negative)
   * @return A qubit reference
   *
   * @par Example:
   * ```c++
   * auto q0 = builder.staticQubit(0);
   * ```
   * ```mlir
   * %q0 = quartz.static 0 : !quartz.qubit
   * ```
   */
  Value staticQubit(int64_t index);

  /**
   * @brief Allocate a qubit register
   * @param size Number of qubits (must be positive)
   * @param name Register name (default: "q")
   * @return Vector of qubit references
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubitRegister(3, "q");
   * ```
   * ```mlir
   * %q0 = quartz.alloc("q", 3, 0) : !quartz.qubit
   * %q1 = quartz.alloc("q", 3, 1) : !quartz.qubit
   * %q2 = quartz.alloc("q", 3, 2) : !quartz.qubit
   * ```
   */
  SmallVector<Value> allocQubitRegister(int64_t size, StringRef name = "q");

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
    StringRef name;
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
   * @return A reference to a ClassicalRegister structure
   *
   * @par Example:
   * ```c++
   * auto c = builder.allocClassicalBitRegister(3, "c");
   * ```
   */
  ClassicalRegister& allocClassicalBitRegister(int64_t size,
                                               StringRef name = "c");

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measure a qubit in the computational basis
   *
   * @details
   * Measures a qubit in place and returns the classical measurement result.
   *
   * @param qubit The qubit to measure
   * @return Classical measurement result (i1)
   *
   * @par Example:
   * ```c++
   * auto result = builder.measure(q);
   * ```
   * ```mlir
   * %result = quartz.measure %q : !quartz.qubit -> i1
   * ```
   */
  Value measure(Value qubit);

  /**
   * @brief Measure a qubit and store the result in a bit of a register
   *
   * @param qubit The qubit to measure
   * @param bit The classical bit to store the result
   *
   * @par Example:
   * ```c++
   * builder.measure(q0, c[0]);
   * ```
   * ```mlir
   * %r0 = quartz.measure("c", 3, 0) %q0 : !quartz.qubit -> i1
   * ```
   */
  QuartzProgramBuilder& measure(Value qubit, const Bit& bit);

  /**
   * @brief Reset a qubit to |0⟩ state
   *
   * @details
   * Resets a qubit to the |0⟩ state in place.
   *
   * @param qubit The qubit to reset
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.reset(q);
   * ```
   * ```mlir
   * quartz.reset %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& reset(Value qubit);

  //===--------------------------------------------------------------------===//
  // Unitary Operations
  //===--------------------------------------------------------------------===//

  // ZeroTargetOneParameter

#define DECLARE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)            \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM);                                                   \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.OP_NAME(%PARAM)                                                    \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM));    \
  /**                                                                          \
   * Apply a controlled OP_CLASS                                               \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Control qubit                                              \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM, {q0, q1});                                      \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME(%PARAM)                                                  \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM),  \
                                   Value control);                             \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Control qubits                                            \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM, {q0, q1});                                     \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME(%PARAM)                                                  \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM), \
                                    ValueRange controls);

  DECLARE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DECLARE_ZERO_TARGET_ONE_PARAMETER

  // OneTargetZeroParameter

#define DECLARE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                   \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @param qubit Target qubit                                                 \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(q);                                                       \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.OP_NAME %q : !quartz.qubit                                         \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(Value qubit);                                  \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
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
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME %q1 : !quartz.qubit                                      \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(Value control, Value target);               \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
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
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME %q2 : !quartz.qubit                                      \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(ValueRange controls, Value target);

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
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
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
   * quartz.OP_NAME(%PARAM) %q : !quartz.qubit                                 \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM),     \
                                Value qubit);                                  \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
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
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME(%PARAM) %q1 : !quartz.qubit                              \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM),  \
                                   Value control, Value target);               \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
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
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME(%PARAM) %q2 : !quartz.qubit                              \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM), \
                                    ValueRange controls, Value target);

  DECLARE_ONE_TARGET_ONE_PARAMETER(RzOp, rz, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(RxOp, rx, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(RyOp, ry, theta)
  DECLARE_ONE_TARGET_ONE_PARAMETER(POp, p, theta)

#undef DECLARE_ONE_TARGET_ONE_PARAMETER

  // OneTargetTwoParameter

#define DECLARE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)    \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
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
   * quartz.OP_NAME(%PARAM1, %PARAM2) %q : !quartz.qubit                       \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM1),    \
                                const std::variant<double, Value>&(PARAM2),    \
                                Value qubit);                                  \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
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
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME(%PARAM1, %PARAM2) %q1 : !quartz.qubit                    \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   Value control, Value target);               \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
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
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME(%PARAM1, %PARAM2) %q2 : !quartz.qubit                    \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(                                           \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value target);

  DECLARE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
  DECLARE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DECLARE_ONE_TARGET_TWO_PARAMETER

  // OneTargetThreeParameter

#define DECLARE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,  \
                                           PARAM3)                             \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
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
   * quartz.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q : !quartz.qubit              \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM1),    \
                                const std::variant<double, Value>&(PARAM2),    \
                                const std::variant<double, Value>&(PARAM3),    \
                                Value qubit);                                  \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
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
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q1 : !quartz.qubit           \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   const std::variant<double, Value>&(PARAM3), \
                                   Value control, Value target);               \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
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
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q2 : !quartz.qubit           \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(                                           \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), ValueRange controls,         \
      Value target);

  DECLARE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DECLARE_ONE_TARGET_THREE_PARAMETER

  // TwoTargetZeroParameter

#define DECLARE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                   \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
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
   * quartz.OP_NAME %q0, %q1 : !quartz.qubit, !quartz.qubit                    \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(Value qubit0, Value qubit1);                   \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @param control Control qubit                                              \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(q0, q1, q2);                                           \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME %q1, %q2 : !quartz.qubit, !quartz.qubit                  \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(Value control, Value qubit0, Value qubit1); \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @param controls Control qubits                                            \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME({q0, q1}, q2, q3);                                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME %q2, %q3 : !quartz.qubit, !quartz.qubit                  \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(ValueRange controls, Value qubit0,         \
                                    Value qubit1);

  DECLARE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
  DECLARE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DECLARE_TWO_TARGET_ZERO_PARAMETER

  // TwoTargetOneParameter

#define DECLARE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
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
   * quartz.OP_NAME(%PARAM) %q0, %q1 : !quartz.qubit, !quartz.qubit            \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM),     \
                                Value qubit0, Value qubit1);                   \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Control qubit                                              \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM, q0, q1, q2);                                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME(%PARAM) %q1, %q2 : !quartz.qubit, !quartz.qubit          \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM),  \
                                   Value control, Value qubit0, Value qubit1); \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Control qubits                                            \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM, {q0, q1}, q2, q3);                             \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0, %q1) {                                                   \
   *   quartz.OP_NAME(%PARAM) %q2, %q3 : !quartz.qubit, !quartz.qubit          \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(const std::variant<double, Value>&(PARAM), \
                                    ValueRange controls, Value qubit0,         \
                                    Value qubit1);

  DECLARE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
  DECLARE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DECLARE_TWO_TARGET_ONE_PARAMETER

  // TwoTargetTwoParameter

#define DECLARE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)    \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
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
   * quartz.OP_NAME(%PARAM1, %PARAM2) %q0, %q1 : !quartz.qubit, !quartz.qubit  \
   * ```                                                                       \
   */                                                                          \
  QuartzProgramBuilder& OP_NAME(const std::variant<double, Value>&(PARAM1),    \
                                const std::variant<double, Value>&(PARAM2),    \
                                Value qubit0, Value qubit1);                   \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param control Control qubit                                              \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.c##OP_NAME(PARAM1, PARAM2, q0, q1, q2);                           \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0) {                                                        \
   *   quartz.OP_NAME(%PARAM1, %PARAM2) %q1, %q2 : !quartz.qubit,              \
   * !quartz.qubit                                                             \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& c##OP_NAME(const std::variant<double, Value>&(PARAM1), \
                                   const std::variant<double, Value>&(PARAM2), \
                                   Value control, Value qubit0, Value qubit1); \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param controls Control qubits                                            \
   * @param qubit0 Target qubit                                                \
   * @param qubit1 Target qubit                                                \
   * @return Reference to this builder for method chaining                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.mc##OP_NAME(PARAM1, PARAM2, {q0, q1}, q2, q3);                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * quartz.ctrl(%q0, %q1) {                                                   \
   *  quartz.OP_NAME(%PARAM1, %PARAM2) %q2, %q3 : !quartz.qubit, !quartz.qubit \
   * }                                                                         \
   */                                                                          \
  QuartzProgramBuilder& mc##OP_NAME(                                           \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), ValueRange controls,         \
      Value qubit0, Value qubit1);

  DECLARE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
  DECLARE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DECLARE_TWO_TARGET_TWO_PARAMETER

  // BarrierOp

  /**
   * @brief Apply a BarrierOp
   *
   * @param qubits Target qubits
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.barrier({q0, q1});
   * ```
   * ```mlir
   * quartz.barrier %q0, %q1 : !quartz.qubit, !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& barrier(ValueRange qubits);

  //===--------------------------------------------------------------------===//
  // Modifiers
  //===--------------------------------------------------------------------===//

  /**
   * @brief Apply a controlled operation
   *
   * @param controls Control qubits
   * @param body Function that builds the body containing the target operation
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.ctrl(q0, [&](auto& b) { b.x(q1); });
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.x %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& ctrl(ValueRange controls,
                             const std::function<void(OpBuilder&)>& body);

  //===--------------------------------------------------------------------===//
  // Deallocation
  //===--------------------------------------------------------------------===//

  /**
   * @brief Explicitly deallocate a qubit
   *
   * @details
   * Deallocates a qubit and removes it from tracking. Optional, finalize()
   * automatically deallocates all remaining allocated qubits.
   *
   * @param qubit The qubit to deallocate
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.dealloc(q);
   * ```
   * ```mlir
   * quartz.dealloc %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& dealloc(Value qubit);

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Finalize the program and return the constructed module
   *
   * @details
   * Automatically deallocates all remaining allocated qubits, adds a return
   * statement with exit code 0 (indicating successful execution), and
   * transfers ownership of the module to the caller.
   * The builder should not be used after calling this method.
   *
   * @return OwningOpRef containing the constructed quantum program module
   */
  OwningOpRef<ModuleOp> finalize();

private:
  MLIRContext* ctx{};
  Location loc;
  ModuleOp module;

  /// Track allocated qubits for automatic deallocation
  llvm::DenseSet<Value> allocatedQubits;

  /// Track allocated classical Registers
  SmallVector<ClassicalRegister> allocatedClassicalRegisters;

  /// Check if the builder has been finalized
  void checkFinalized() const;
};
} // namespace mlir::quartz
