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

#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <cstdint>
#include <functional>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <string>
#include <utility>
#include <variant>

namespace mlir::qco {

/**
 * @brief Builder API for constructing quantum programs in the QCO dialect
 *
 * @details
 * The QCOProgramBuilder provides a type-safe interface for constructing
 * quantum circuits using value semantics. Operations consume input qubit
 * SSA values and produce new output values, following the functional
 * programming paradigm.
 *
 * @par Linear Type Enforcement:
 * The builder enforces linear type semantics by tracking valid qubit SSA
 * values. Once a qubit is consumed by an operation producing a new version
 * (e.g., reset, measure), the old SSA value is invalidated. This prevents
 * use-after-consume errors and mirrors quantum computing's no-cloning theorem.
 *
 * @par Example Usage:
 * ```c++
 * QCOProgramBuilder builder(context);
 * builder.initialize();
 *
 * auto q0 = builder.staticQubit(0);
 * auto q1 = builder.staticQubit(1);
 *
 * // Operations return updated values
 * q0 = builder.h(q0);
 * std::tie(q0, q1) = builder.cx(q0, q1);
 *
 * auto module = builder.finalize();
 * ```
 */
class QCOProgramBuilder final : public OpBuilder {
public:
  /**
   * @brief Construct a new QCOProgramBuilder
   * @param context The MLIR context to use for building operations
   */
  explicit QCOProgramBuilder(MLIRContext* context);

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
   * @return A tracked, valid qubit SSA value
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubit();
   * ```
   * ```mlir
   * %q = qco.alloc : !qco.qubit
   * ```
   */
  Value allocQubit();

  /**
   * @brief Get a static qubit by index
   * @param index The qubit index (must be non-negative)
   * @return A tracked, valid qubit SSA value
   *
   * @par Example:
   * ```c++
   * auto q0 = builder.staticQubit(0);
   * ```
   * ```mlir
   * %q0 = qco.static 0 : !qco.qubit
   * ```
   */
  Value staticQubit(int64_t index);

  /**
   * @brief Allocate a qubit register
   * @param size Number of qubits (must be positive)
   * @param name Register name (default: "q")
   * @return Vector of tracked, valid qubit SSA values
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubitRegister(3, "q");
   * ```
   * ```mlir
   * %q0 = qco.alloc("q", 3, 0) : !qco.qubit
   * %q1 = qco.alloc("q", 3, 1) : !qco.qubit
   * %q2 = qco.alloc("q", 3, 2) : !qco.qubit
   * ```
   */
  llvm::SmallVector<Value> allocQubitRegister(int64_t size,
                                              const std::string& name = "q");

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
        const std::string msg = "Bit index " + std::to_string(index) +
                                " out of bounds for register '" + name +
                                "' of size " + std::to_string(size);
        llvm::reportFatalUsageError(msg.c_str());
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
  [[nodiscard]] ClassicalRegister
  allocClassicalBitRegister(int64_t size, std::string name = "c") const;

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measure a qubit in the computational basis
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value
   * along with the measurement result (i1). The input is validated and
   * tracking is updated to reflect the new output value.
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Pair of (output_qubit, measurement_result)
   *
   * @par Example:
   * ```c++
   * auto [q_out, result] = builder.measure(q);
   * ```
   * ```mlir
   * %q_out, %result = qco.measure %q : !qco.qubit
   * ```
   */
  std::pair<Value, Value> measure(Value qubit);

  /**
   * @brief Measure a qubit and record the result in a bit of a register
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @param bit The classical bit to record the result
   * @return Output qubit value
   *
   * @par Example:
   * ```c++
   * q0 = builder.measure(q0, c[0]);
   * ```
   * ```mlir
   * %q0_out, %r0 = qco.measure("c", 3, 0) %q0 : !qco.qubit
   * ```
   */
  Value measure(Value qubit, const Bit& bit);

  /**
   * @brief Reset a qubit to |0⟩ state
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value
   * in the |0⟩ state. The input is validated and tracking is updated.
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit value
   *
   * @par Example:
   * ```c++
   * q = builder.reset(q);
   * ```
   * ```mlir
   * %q_out = qco.reset %q : !qco.qubit -> !qco.qubit
   * ```
   */
  Value reset(Value qubit);

  //===--------------------------------------------------------------------===//
  // Unitary Operations
  //===--------------------------------------------------------------------===//

  // ZeroTargetOneParameter

#define DECLARE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)            \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * builder.OP_NAME(PARAM);                                                   \
   * ```                                                                       \
   * ```mlir                                                                   \
   * qco.OP_NAME(%PARAM)                                                       \
   * ```                                                                       \
   */                                                                          \
  void OP_NAME(const std::variant<double, Value>&(PARAM));                     \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Input control qubit                                        \
   * @return Output control qubit                                              \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * q_out = builder.c##OP_NAME(PARAM, q_in);                                  \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q_out = qco.ctrl(%q_in) {                                                \
   *   qco.OP_NAME(%PARAM)                                                     \
   *   qco.yield                                                               \
   * } : ({!qco.qubit}) -> ({!qco.qubit})                                      \
   * ```                                                                       \
   */                                                                          \
  Value c##OP_NAME(const std::variant<double, Value>&(PARAM), Value control);  \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Control qubits                                            \
   * @return Output control qubits                                             \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.mc##OP_NAME(PARAM, {q0_in, q1_in});            \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.ctrl(%q0_in, %q1_in) {                             \
   *   qco.OP_NAME(%PARAM)                                                     \
   *   qco.yield                                                               \
   * } : ({!qco.qubit, !qco.qubit}) -> ({!qco.qubit, !qco.qubit})              \
   * ```                                                                       \
   */                                                                          \
  ValueRange mc##OP_NAME(const std::variant<double, Value>&(PARAM),            \
                         ValueRange controls);

  DECLARE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DECLARE_ZERO_TARGET_ONE_PARAMETER

  // OneTargetZeroParameter

#define DECLARE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                   \
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input qubit and produces a new output qubit SSA value. The   \
   * input is validated and the tracking is updated.                           \
   *                                                                           \
   * @param qubit Input qubit (must be valid/unconsumed)                       \
   * @return Output qubit                                                      \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * q_out = builder.OP_NAME(q_in);                                            \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME %q_in : !qco.qubit -> !qco.qubit                     \
   * ```                                                                       \
   */                                                                          \
  Value OP_NAME(Value qubit);                                                  \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubit, output_target_qubit)               \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.c##OP_NAME(q0_in, q1_in);                      \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.ctrl(%q0_in) %q1_in {                              \
   *   %q1_res = qco.OP_NAME %q1_in : !qco.qubit -> !qco.qubit                 \
   *   qco.yield %q1_res                                                       \
   * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})          \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> c##OP_NAME(Value control, Value target);             \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubits, output_target_qubit)              \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, target_out} = builder.mc##OP_NAME({q0_in, q1_in}, q2_in);  \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %target_out = qco.ctrl(%q0_in, %q1_in) %q2_in {            \
   *   %q2_res = qco.OP_NAME %q2_in : !qco.qubit -> !qco.qubit                 \
   *   qco.yield %q2_res                                                       \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit,             \
   * !qco.qubit}, {!qco.qubit})                                                \
   * ```                                                                       \
   */                                                                          \
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
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input qubit and produces a new output qubit SSA value. The   \
   * input is validated and the tracking is updated.                           \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param qubit Input qubit (must be valid/unconsumed)                       \
   * @return Output qubit                                                      \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * q_out = builder.OP_NAME(PARAM, q_in);                                     \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM) %q_in : !qco.qubit -> !qco.qubit             \
   * ```                                                                       \
   */                                                                          \
  Value OP_NAME(const std::variant<double, Value>&(PARAM), Value qubit);       \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubit, output_target_qubit)               \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.c##OP_NAME(PARAM, q0_in, q1_in);               \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.ctrl(%q0_in) %q1_in {                              \
   *   %q1_res = qco.OP_NAME(%PARAM) %q1_in : !qco.qubit -> !qco.qubit         \
   *   qco.yield %q1_res                                                       \
   * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})          \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> c##OP_NAME(                                          \
      const std::variant<double, Value>&(PARAM), Value control, Value target); \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubits, output_target_qubit)              \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, target_out} = builder.mc##OP_NAME(PARAM, {q0_in, q1_in},   \
   * q2_in);                                                                   \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %target_out = qco.ctrl(%q0_in, %q1_in) %q2_in {            \
   *   %q2_res = qco.OP_NAME(%PARAM) %q2_in : !qco.qubit -> !qco.qubit         \
   *   qco.yield %q2_res                                                       \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit,             \
   * !qco.qubit}, {!qco.qubit})                                                \
   * ```                                                                       \
   */                                                                          \
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
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input qubit and produces a new output qubit SSA value. The   \
   * input is validated and the tracking is updated.                           \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param qubit Input qubit (must be valid/unconsumed)                       \
   * @return Output qubit                                                      \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * q_out = builder.OP_NAME(PARAM1, PARAM2, q_in);                            \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM1, %PARAM2) %q_in : !qco.qubit ->              \
   * !qco.qubit                                                                \
   * ```                                                                       \
   */                                                                          \
  Value OP_NAME(const std::variant<double, Value>&(PARAM1),                    \
                const std::variant<double, Value>&(PARAM2), Value qubit);      \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubit, output_target_qubit)               \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.c##OP_NAME(PARAM1, PARAM2, q0_in, q1_in);      \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.ctrl(%q0_in) %q1_in {                              \
   *   %q1_res = qco.OP_NAME(%PARAM1, %PARAM2) %q1_in : !qco.qubit ->          \
   * !qco.qubit                                                                \
   *   qco.yield %q1_res                                                       \
   * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})          \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> c##OP_NAME(                                          \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control,               \
      Value target);                                                           \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubits, output_target_qubit)              \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, target_out} = builder.mc##OP_NAME(PARAM1, PARAM2, {q0_in,  \
   * q1_in}, q2_in);                                                           \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %target_out = qco.ctrl(%q0_in, %q1_in) %q2_in {            \
   *   %q2_res = qco.OP_NAME(%PARAM1, %PARAM2) %q2_in : !qco.qubit ->          \
   * !qco.qubit                                                                \
   *   qco.yield %q2_res                                                       \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit,             \
   * !qco.qubit}, {!qco.qubit})                                                \
   * ```                                                                       \
   */                                                                          \
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
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input qubit and produces a new output qubit SSA value. The   \
   * input is validated and the tracking is updated.                           \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param PARAM3 Rotation angle in radians                                   \
   * @param qubit Input qubit (must be valid/unconsumed)                       \
   * @return Output qubit                                                      \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * q_out = builder.OP_NAME(PARAM1, PARAM2, PARAM3, q_in);                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q_in : !qco.qubit ->     \
   * !qco.qubit                                                                \
   * ```                                                                       \
   */                                                                          \
  Value OP_NAME(const std::variant<double, Value>&(PARAM1),                    \
                const std::variant<double, Value>&(PARAM2),                    \
                const std::variant<double, Value>&(PARAM3), Value qubit);      \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param PARAM3 Rotation angle in radians                                   \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubit, output_target_qubit)               \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.c##OP_NAME(PARAM1, PARAM2, PARAM3, q0_in,      \
   * q1_in);                                                                   \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.ctrl(%q0_in) %q1_in {                              \
   *   %q1_res = qco.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q1_in : !qco.qubit    \
   * -> !qco.qubit                                                             \
   *   qco.yield %q1_res                                                       \
   * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})          \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> c##OP_NAME(                                          \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2),                              \
      const std::variant<double, Value>&(PARAM3), Value control,               \
      Value target);                                                           \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param PARAM3 Rotation angle in radians                                   \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param target Input target qubit (must be valid/unconsumed)               \
   * @return Pair of (output_control_qubits, output_target_qubit)              \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, target_out} = builder.mc##OP_NAME(PARAM1, PARAM2, PARAM3,  \
   * {q0_in, q1_in}, q2_in);                                                   \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %target_out = qco.ctrl(%q0_in, %q1_in) %q2_in {            \
   *   %q2_res = qco.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q2_in : !qco.qubit    \
   * -> !qco.qubit                                                             \
   *   qco.yield %q2_res                                                       \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit,             \
   * !qco.qubit}, {!qco.qubit})                                                \
   * ```                                                                       \
   */                                                                          \
  std::pair<ValueRange, Value> mc##OP_NAME(                                    \
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
   * @details                                                                  \
   * Consumes the input qubits and produces new output qubit SSA values. The   \
   * inputs are validated and the tracking is updated.                         \
   *                                                                           \
   * @param qubit0 Input qubit (must be valid/unconsumed)                      \
   * @param qubit1 Input qubit (must be valid/unconsumed)                      \
   * @return Output qubits                                                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.OP_NAME(q0_in, q1_in);                         \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME %q0_in, %q1_in : !qco.qubit, !qco.qubit    \
   * -> !qco.qubit, !qco.qubit                                                 \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> OP_NAME(Value qubit0, Value qubit1);                 \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param qubit0 Target qubit (must be valid/unconsumed)                     \
   * @param qubit1 Target qubit (must be valid/unconsumed)                     \
   * @return Pair of (output_control_qubit, (output_qubit0, output_qubit1))    \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, {q1_out, q2_out}} = builder.c##OP_NAME(q0_in, q1_in, q2_in);     \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out, %q2_out = qco.ctrl(%q0_in) %q1_in, %q2_in {             \
   *   %q1_res, %q2_res = qco.OP_NAME %q1_in, %q2_in : !qco.qubit,             \
   * !qco.qubit -> !qco.qubit, !qco.qubit                                      \
   *   qco.yield %q1_res, %q2_res                                              \
   * } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit},            \
   * {!qco.qubit, !qco.qubit})                                                 \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, std::pair<Value, Value>> c##OP_NAME(                        \
      Value control, Value qubit0, Value qubit1);                              \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param qubit0 Target qubit (must be valid/unconsumed)                     \
   * @param qubit1 Target qubit (must be valid/unconsumed)                     \
   * @return Pair of (output_control_qubits, (output_qubit0, output_qubit1))   \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, {q1_out, q2_out}} = builder.mc##OP_NAME({q0_in, q1_in},    \
   * q2_in, q3_in);                                                            \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %q1_out, %q2_out = qco.ctrl(%q0_in, %q1_in) %q2_in,        \
   * %q3_in {                                                                  \
   *   %q2_res, %q3_res = qco.OP_NAME %q2_in, %q3_in : !qco.qubit,             \
   * !qco.qubit -> !qco.qubit, !qco.qubit                                      \
   *   qco.yield %q2_res, %q3_res                                              \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit}) ->               \
   * ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit})                      \
   * ```                                                                       \
   */                                                                          \
  std::pair<ValueRange, std::pair<Value, Value>> mc##OP_NAME(                  \
      ValueRange controls, Value qubit0, Value qubit1);

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
   * @details                                                                  \
   * Consumes the input qubits and produces new output qubit SSA values. The   \
   * inputs are validated and the tracking is updated.                         \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param qubit0 Input qubit (must be valid/unconsumed)                      \
   * @param qubit1 Input qubit (must be valid/unconsumed)                      \
   * @return Output qubits                                                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.OP_NAME(PARAM, q0_in, q1_in);                  \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME(%PARAM) %q0_in, %q1_in : !qco.qubit,       \
   * !qco.qubit                                                                \
   * -> !qco.qubit, !qco.qubit                                                 \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> OP_NAME(const std::variant<double, Value>&(PARAM),   \
                                  Value qubit0, Value qubit1);                 \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param qubit0 Target qubit (must be valid/unconsumed)                     \
   * @param qubit1 Target qubit (must be valid/unconsumed)                     \
   * @return Pair of (output_control_qubit, (output_qubit0, output_qubit1))    \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, {q1_out, q2_out}} = builder.c##OP_NAME(PARAM, q0_in, q1_in,      \
   * q2_in);                                                                   \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out, %q2_out = qco.ctrl(%q0_in) %q1_in, %q2_in {             \
   *   %q1_res, %q2_res = qco.OP_NAME(%PARAM) %q1_in, %q2_in : !qco.qubit,     \
   * !qco.qubit -> !qco.qubit, !qco.qubit                                      \
   *   qco.yield %q1_res, %q2_res                                              \
   * } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit},            \
   * {!qco.qubit, !qco.qubit})                                                 \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, std::pair<Value, Value>> c##OP_NAME(                        \
      const std::variant<double, Value>&(PARAM), Value control, Value qubit0,  \
      Value qubit1);                                                           \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM Rotation angle in radians                                    \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param qubit0 Target qubit (must be valid/unconsumed)                     \
   * @param qubit1 Target qubit (must be valid/unconsumed)                     \
   * @return Pair of (output_control_qubits, (output_qubit0, output_qubit1))   \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, {q1_out, q2_out}} = builder.mc##OP_NAME(PARAM, {q0_in,     \
   * q1_in}, q2_in, q3_in);                                                    \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %q1_out, %q2_out = qco.ctrl(%q0_in, %q1_in) %q2_in,        \
   * %q3_in {                                                                  \
   *   %q2_res, %q3_res = qco.OP_NAME(%PARAM) %q2_in, %q3_in : !qco.qubit,     \
   * !qco.qubit -> !qco.qubit, !qco.qubit                                      \
   *   qco.yield %q2_res, %q3_res                                              \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit}) ->               \
   * ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit})                      \
   * ```                                                                       \
   */                                                                          \
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
  /**                                                                          \
   * @brief Apply a OP_CLASS                                                   \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input qubits and produces new output qubit SSA values. The   \
   * inputs are validated and the tracking is updated.                         \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param qubit0 Input qubit (must be valid/unconsumed)                      \
   * @param qubit1 Input qubit (must be valid/unconsumed)                      \
   * @return Output qubits                                                     \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, q1_out} = builder.OP_NAME(PARAM1, PARAM2, q0_in, q1_in);         \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME(%PARAM1, %PARAM2) %q0_in, %q1_in :         \
   * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit                          \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, Value> OP_NAME(const std::variant<double, Value>&(PARAM1),  \
                                  const std::variant<double, Value>&(PARAM2),  \
                                  Value qubit0, Value qubit1);                 \
  /**                                                                          \
   * @brief Apply a controlled OP_CLASS                                        \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param control Input control qubit (must be valid/unconsumed)             \
   * @param qubit0 Target qubit (must be valid/unconsumed)                     \
   * @param qubit1 Target qubit (must be valid/unconsumed)                     \
   * @return Pair of (output_control_qubit, (output_qubit0, output_qubit1))    \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {q0_out, {q1_out, q2_out}} = builder.c##OP_NAME(PARAM1, PARAM2, q0_in,    \
   * q1_in, q2_in);                                                            \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %q0_out, %q1_out, %q2_out = qco.ctrl(%q0_in) %q1_in, %q2_in {             \
   *   %q1_res, %q2_res = qco.OP_NAME(%PARAM1, %PARAM2) %q1_in, %q2_in :       \
   * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit                          \
   *   qco.yield %q1_res, %q2_res                                              \
   * } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) ->                           \
   * ({!qco.qubit}, {!qco.qubit, !qco.qubit})                                  \
   * ```                                                                       \
   */                                                                          \
  std::pair<Value, std::pair<Value, Value>> c##OP_NAME(                        \
      const std::variant<double, Value>&(PARAM1),                              \
      const std::variant<double, Value>&(PARAM2), Value control, Value qubit0, \
      Value qubit1);                                                           \
  /**                                                                          \
   * @brief Apply a multi-controlled OP_CLASS                                  \
   *                                                                           \
   * @details                                                                  \
   * Consumes the input control and target qubits and produces new output      \
   * qubit SSA values. The inputs are validated and the tracking is updated.   \
   *                                                                           \
   * @param PARAM1 Rotation angle in radians                                   \
   * @param PARAM2 Rotation angle in radians                                   \
   * @param controls Input control qubits (must be valid/unconsumed)           \
   * @param qubit0 Target qubit (must be valid/unconsumed)                     \
   * @param qubit1 Target qubit (must be valid/unconsumed)                     \
   * @return Pair of (output_control_qubits, (output_qubit0, output_qubit1))   \
   *                                                                           \
   * @par Example:                                                             \
   * ```c++                                                                    \
   * {controls_out, {q1_out, q2_out}} = builder.mc##OP_NAME(PARAM1, PARAM2,    \
   * {q0_in, q1_in}, q2_in, q3_in);                                            \
   * ```                                                                       \
   * ```mlir                                                                   \
   * %controls_out, %q1_out, %q2_out = qco.ctrl(%q0_in, %q1_in) %q2_in,        \
   * %q3_in {                                                                  \
   *   %q2_res, %q3_res = qco.OP_NAME(%PARAM1, %PARAM2) %q2_in, %q3_in :       \
   * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit                          \
   *   qco.yield %q2_res, %q3_res                                              \
   * } : ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit}) ->               \
   * ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit})                      \
   * ```                                                                       \
   */                                                                          \
  std::pair<ValueRange, std::pair<Value, Value>> mc##OP_NAME(                  \
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
   * @param qubits Input qubits (must be valid/unconsumed)
   * @return Output qubits
   *
   * @par Example:
   * ```c++
   * builder.barrier({q0, q1});
   * ```
   * ```mlir
   * qco.barrier %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit,
   * !qco.qubit
   * ```
   */
  ValueRange barrier(ValueRange qubits);

  //===--------------------------------------------------------------------===//
  // Modifiers
  //===--------------------------------------------------------------------===//

  /**
   * @brief Apply a controlled operation
   *
   * @param controls Control qubits
   * @param targets Target qubits
   * @param body Function that builds the body containing the target operation
   * @return Pair of (output_control_qubits, output_target_qubits)
   *
   * @par Example:
   * ```c++
   * {controls_out, targets_out} =
   *   builder.ctrl(q0_in, q1_in, [&](ValueRange targets) {
   *     auto q1_res = builder.x(targets[0]);
   *     return {q1_res};
   *   });
   * ```
   * ```mlir
   * %controls_out, %targets_out = qco.ctrl(%q0_in) %q1_in {
   *   %q1_res = qco.x %q1_in : !qco.qubit -> !qco.qubit
   *   qco.yield %q1_res
   * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
   * ```
   */
  std::pair<ValueRange, ValueRange>
  ctrl(ValueRange controls, ValueRange targets,
       const std::function<ValueRange(ValueRange)>& body);

  //===--------------------------------------------------------------------===//
  // Deallocation
  //===--------------------------------------------------------------------===//

  /**
   * @brief Explicitly deallocate a qubit
   *
   * @details
   * Validates and removes the qubit from tracking. Optional, finalize()
   * automatically deallocates all remaining qubits.
   *
   * @param qubit Qubit to deallocate (must be valid/unconsumed)
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.dealloc(q);
   * ```
   * ```mlir
   * qco.dealloc %q : !qco.qubit
   * ```
   */
  QCOProgramBuilder& dealloc(Value qubit);

  //===--------------------------------------------------------------------===//
  // SCF operations
  //===--------------------------------------------------------------------===//

  /**
   * @brief Constructs a scf.for operation with iterArgs
   *
   * @param lowerbound Lowerbound of the loop
   * @param upperbound Upperbound of the loop
   * @param step Stepsize of the loop
   * @param initArgs Initial arguments for the iterArgs
   * @param body Function that builds the body of the for operation
   * @return ValueRange of the results
   *
   * @par Example:
   * ```c++
   * builder.scfFor(lb, ub, step, initArgs, [&] {
   *    auto q1 = builder.x(initArgs[0]);
   *    builder.scfYield(q1);
   * });
   * ```
   * ```mlir
   * %q1 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %q0)
   * -> !qco.qubit {
   *   %q2 = qc.x %arg0 : !qco.qubit -> !qco.qubit
   *   scf.yield %q2 : !qco.qubit
   * }
   * ```
   */
  ValueRange scfFor(Value lowerbound, Value upperbound, Value step,
                    ValueRange initArgs,
                    const std::function<ValueRange(Value, ValueRange)>& body);
  /**
   * @brief Constructs a scf.while operation with return values
   *
   * @param args Arguments for the while loop
   * @param beforeBody Function that builds the before body of the while
   * operation
   * @param afterBody Function that builds the after body of the while operation
   * @return ValueRange of the results
   *
   * @par Example:
   * ```c++
   * builder.scfWhile(args, [&](ValueRange iterArgs) {
   *   auto q1 = builder.h(iterArgs[0]);
   *   auto [q2, measureRes] = builder.measure(q1);
   *   builder.condition(measureRes, q2);
   * }, [&](ValueRange iterArgs) {
   *   auto q1 = builder.x(iterArgs[0]);
   *   builder.scfYield(q1);
   * });
   * ```
   * ```mlir
   * %q1 = scf.while (%arg0 = %q0): (!qco.qubit) -> (!qco.qubit) {
   *   %q2 = qco.h(%arg0)
   *   %q3, %result = qco.measure %q2 : !qco.qubit
   *   scf.condition(%result) %q3 : !qco.qubit
   * } do {
   * ^bb0(%arg0 : !qco.qubit):
   *   %q4 = qco.x %arg0 : !qco.qubit -> !qco.qubit
   *   scf.yield %q4 : !qco.qubit
   * }
   * ```
   */
  ValueRange scfWhile(ValueRange args,
                      const std::function<ValueRange(ValueRange)>& beforeBody,
                      const std::function<ValueRange(ValueRange)>& afterBody);

  /**
   * @brief Constructs a scf.if operation with return values
   *
   * @param condition Condition for the if operation
   * @param qubits Qubits used in the if/else body
   * @param thenBody Function that builds the then body of the if
   * operation
   * @param elseBody Function that builds the else body of the if operation
   * @return ValueRange of the results
   *
   * @par Example:
   * ```c++
   * builder.scfIf(condition, qubits, [&] {
   *   auto q1 = builder.h(q0);
   *   builder.scfYield(q1);
   * }, [&] {
   *   auto q1 = builder.x(q0);
   *   builder.scfYield(q1);
   * });
   * ```
   * ```mlir
   * %q1 = scf.if %condition -> (!qco.qubit) {
   *   %q2 = qco.h %q0 : !qco.qubit -> !qco.qubit
   *   scf.yield %q2 : !qco.qubit
   * } else {
   *   %q2 = qco.x %q0 : !qco.qubit -> !qco.qubit
   *   scf.yield %q2 : !qco.qubit
   * }
   * ```
   */
  ValueRange scfIf(Value condition, ValueRange qubits,
                   const std::function<ValueRange()>& thenBody,
                   const std::function<ValueRange()>& elseBody);

  /**
   * @brief Constructs a scf.condition operation with yielded values
   *
   * @param condition Condition for condition operation
   * @param yieldedValues ValueRange of the yieldedValues
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.scfCondition(condition, yieldedValues);
   * ```
   * ```mlir
   * scf.condition(%condition) %q0 : !qco.qubit
   * ```
   */
  QCOProgramBuilder& scfCondition(Value condition, ValueRange yieldedValues);

  /**
   * @brief Constructs a scf.yield operation with yielded values
   *
   * @param yieldedValues ValueRange of the yieldedValues
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.scfYield(yieldedValues);
   * ```
   * ```mlir
   * scf.yield %q0 : !qco.qubit
   * ```
   */
  QCOProgramBuilder& scfYield(ValueRange yieldedValues);

  //===--------------------------------------------------------------------===//
  // Func operations
  //===--------------------------------------------------------------------===//

  /**
   * @brief Constructs a func.return operation with return values
   *
   * @param returnValues ValueRange of the returned values
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.funcReturn(yieldedValues);
   * ```
   * ```mlir
   * func.return %q0 : !qco.qubit
   * ```
   */
  QCOProgramBuilder& funcReturn(ValueRange returnValues);

  /**
   * @brief Constructs a func.call operation with return values
   *
   * @param name Name of the function that is called
   * @param operands ValueRange of the used operands
   * @return ValueRange of the results
   *
   * @par Example:
   * ```c++
   * auto q1 = builder.funcCall("test", {q0});
   * ```
   * ```mlir
   * %q1 = func.call @test(%q0) : (!qco.qubit) -> !qco.qubit
   * ```
   */
  ValueRange funcCall(StringRef name, ValueRange operands);

  /**
   * @brief Constructs a func.func operation with return values
   *
   * @param name Name of the function that is called
   * @param argTypes TypeRange of the arguments
   * @param resultTypes TypeRange of the results
   * @param body Body of the function
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.funcFunc("test", argTypes, resultTypes, [&](ValueRange args) {
   *   auto q1 = builder.h(args[0]);
   *   builder.funcReturn(q1);
   * })
   * ```
   * ```mlir
   * func.func @test(%arg0 : !qco.qubit) -> !qco.qubit {
   *   %q1 = qco.h %arg0 : !qco.qubit -> !qco.qubit
   *   func.return %q1 : !qco.qubit
   * }
   * ```
   */
  QCOProgramBuilder& funcFunc(StringRef name, TypeRange argTypes,
                              TypeRange resultTypes,
                              const std::function<void(ValueRange)>& body);

  //===--------------------------------------------------------------------===//
  // Arith operations
  //===--------------------------------------------------------------------===//

  /**
   * @brief Constructs a arith.constant of type Index with a given value
   *
   * @param index Value of the constant operation
   * @return Result of the constant operation
   *
   * @par Example:
   * ```c++
   * builder.arithConstantIndex(4);
   * ```
   * ```mlir
   * arith.constant 4 : index
   * ```
   */
  Value arithConstantIndex(int64_t i);

  /**
   * @brief Constructs a arith.constant of type i1 with a given bool value
   *
   * @param b Bool value of the constant operation
   * @return Result of the constant operation
   *
   * @par Example:
   * ```c++
   * builder.arithConstantBool(true);
   * ```
   * ```mlir
   * arith.constant 1 : i1
   * ```
   */
  Value arithConstantBool(bool b);

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Finalize the program and return the constructed module
   *
   * @details
   * Automatically deallocates all remaining valid qubits, adds a return
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

  /// Check if the builder has been finalized
  void checkFinalized() const;

  //===--------------------------------------------------------------------===//
  // Linear Type Tracking Helpers
  //===--------------------------------------------------------------------===//

  /**
   * @brief Validate that a qubit value is valid and unconsumed
   * @param qubit Qubit value to validate
   * @throws Aborts if qubit is not tracked (consumed or never created)
   */
  void validateQubitValue(Value qubit);

  /**
   * @brief Update tracking when an operation consumes and produces a qubit
   * @param inputQubit Input qubit being consumed (must be valid)
   * @param outputQubit New output qubit being produced
   */
  void updateQubitTracking(Value inputQubit, Value outputQubit, Region* region);

  /// Track valid (unconsumed) qubit SSA values for linear type enforcement.
  /// Only values present in this set are valid for use in operations.
  /// When an operation consumes a qubit and produces a new one, the old value
  /// is removed and the new output is added.
  llvm::DenseMap<Region*, llvm::DenseSet<Value>> validQubits;
};
} // namespace mlir::qco
