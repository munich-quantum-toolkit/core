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

#include "mlir/Dialect/Flux/IR/FluxDialect.h"

#include <cstdint>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <variant>
#include <vector>

namespace mlir::flux {

/**
 * @brief Builder API for constructing quantum programs in the Flux dialect
 *
 * @details
 * The FluxProgramBuilder provides a type-safe interface for constructing
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
 * FluxProgramBuilder builder(context);
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
class FluxProgramBuilder final : public OpBuilder {
public:
  /**
   * @brief Construct a new FluxProgramBuilder
   * @param context The MLIR context to use for building operations
   */
  explicit FluxProgramBuilder(MLIRContext* context);

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
   * %q = flux.alloc : !flux.qubit
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
   * %q0 = flux.static 0 : !flux.qubit
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
   * %q0 = flux.alloc("q", 3, 0) : !flux.qubit
   * %q1 = flux.alloc("q", 3, 1) : !flux.qubit
   * %q2 = flux.alloc("q", 3, 2) : !flux.qubit
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
      assert(index < size);
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
   * %q_out, %result = flux.measure %q : !flux.qubit
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
   * %q0_out, %r0 = flux.measure("c", 3, 0) %q0 : !flux.qubit
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
   * %q_out = flux.reset %q : !flux.qubit -> !flux.qubit
   * ```
   */
  Value reset(Value qubit);

  //===--------------------------------------------------------------------===//
  // Unitary Operations
  //===--------------------------------------------------------------------===//

  /**
   * @brief Apply an Id gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit
   *
   * @par Example:
   * ```c++
   * q_out = builder.id(q_in);
   * ```
   * ```mlir
   * %q_out = flux.id %q_in : !flux.qubit -> !flux.qubit
   * ```
   */
  Value id(Value qubit);

  /**
   * @brief Apply a controlled Id gate
   *
   * @param control Input control qubit (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.cid(q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.id %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<Value, Value> cid(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Id gate
   *
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {controls_out, target_out} = builder.mcid({q0_in, q1_in}, q2_in);
   * ```
   * ```mlir
   * %controls_out, %target_out = flux.ctrl(%q0_in, %q1_in) %q2_in {
   *   %q2_res = flux.id %q2_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q2_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit}) -> ({!flux.qubit,
   * !flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, Value> mcid(ValueRange controls, Value target);

  /**
   * @brief Apply an X gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit
   *
   * @par Example:
   * ```c++
   * q_out = builder.x(q_in);
   * ```
   * ```mlir
   * %q_out = flux.x %q_in : !flux.qubit -> !flux.qubit
   * ```
   */
  Value x(Value qubit);

  /**
   * @brief Apply a controlled X gate
   *
   * @param control Input control qubit (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.cx(q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.x %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<Value, Value> cx(Value control, Value target);

  /**
   * @brief Apply a multi-controlled X gate
   *
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {controls_out, target_out} = builder.mcx({q0_in, q1_in}, q2_in);
   * ```
   * ```mlir
   * %controls_out, %target_out = flux.ctrl(%q0_in, %q1_in) %q2_in {
   *   %q2_res = flux.x %q2_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q2_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit}) -> ({!flux.qubit,
   * !flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, Value> mcx(ValueRange controls, Value target);

  /**
   * @brief Apply an S gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit
   *
   * @par Example:
   * ```c++
   * q_out = builder.s(q_in);
   * ```
   * ```mlir
   * %q_out = flux.s %q_in : !flux.qubit -> !flux.qubit
   * ```
   */
  Value s(Value qubit);

  /**
   * @brief Apply a controlled S gate
   *
   * @param control Input control qubit (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.cs(q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.s %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<Value, Value> cs(Value control, Value target);

  /**
   * @brief Apply a multi-controlled S gate
   *
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {controls_out, target_out} = builder.mcs({q0_in, q1_in}, q2_in);
   * ```
   * ```mlir
   * %controls_out, %target_out = flux.ctrl(%q0_in, %q1_in) %q2_in {
   *   %q2_res = flux.s %q2_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q2_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit}) -> ({!flux.qubit,
   * !flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, Value> mcs(ValueRange controls, Value target);

  /**
   * @brief Apply an Sdg gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit
   *
   * @par Example:
   * ```c++
   * q_out = builder.sdg(q_in);
   * ```
   * ```mlir
   * %q_out = flux.sdg %q_in : !flux.qubit -> !flux.qubit
   * ```
   */
  Value sdg(Value qubit);

  /**
   * @brief Apply a controlled Sdg gate
   *
   * @param control Input control qubit (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.csdg(q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.sdg %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<Value, Value> csdg(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Sdg gate
   *
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {controls_out, target_out} = builder.mcsdg({q0_in, q1_in}, q2_in);
   * ```
   * ```mlir
   * %controls_out, %target_out = flux.ctrl(%q0_in, %q1_in) %q2_in {
   *   %q2_res = flux.sdg %q2_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q2_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit}) -> ({!flux.qubit,
   * !flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, Value> mcsdg(ValueRange controls, Value target);

  /**
   * @brief Apply an RX gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param theta Rotation angle in radians
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit
   *
   * @par Example:
   * ```c++
   * q_out = builder.rx(1.0, q_in);
   * ```
   * ```mlir
   * %q_out = flux.rx(1.0) %q_in : !flux.qubit -> !flux.qubit
   * ```
   */
  Value rx(const std::variant<double, Value>& theta, Value qubit);

  /**
   * @brief Apply a controlled RX gate
   *
   * @param theta Rotation angle in radians
   * @param control Input control qubit (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.crx(1.0, q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.rx(1.0) %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<Value, Value> crx(const std::variant<double, Value>& theta,
                              Value control, Value target);

  /**
   * @brief Apply a multi-controlled RX gate
   *
   * @param theta Rotation angle in radians
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {controls_out, target_out} = builder.mcrx(1.0, {q0_in, q1_in}, q2_in);
   * ```
   * ```mlir
   * %controls_out, %target_out = flux.ctrl(%q0_in, %q1_in) %q2_in {
   *   %q2_res = flux.rx(1.0) %q2_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q2_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit}) -> ({!flux.qubit,
   * !flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, Value> mcrx(const std::variant<double, Value>& theta,
                                    ValueRange controls, Value target);

  /**
   * @brief Apply a U2 gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param phi Rotation angle in radians
   * @param lambda Rotation angle in radians
   * @param qubit Input qubit (must be valid/unconsumed)
   * @return Output qubit
   *
   * @par Example:
   * ```c++
   * q_out = builder.u2(1.0, 0.5, q_in);
   * ```
   * ```mlir
   * %q_out = flux.u2(1.0, 0.5) %q_in : !flux.qubit -> !flux.qubit
   * ```
   */
  Value u2(const std::variant<double, Value>& phi,
           const std::variant<double, Value>& lambda, Value qubit);

  /**
   * @brief Apply a controlled U2 gate
   *
   * @param phi Rotation angle in radians
   * @param lambda Rotation angle in radians
   * @param control Input control qubit (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.cu2(1.0, 0.5, q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.u2(1.0, 0.5) %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<Value, Value> cu2(const std::variant<double, Value>& phi,
                              const std::variant<double, Value>& lambda,
                              Value control, Value target);

  /**
   * @brief Apply a multi-controlled U2 gate
   *
   * @param phi Rotation angle in radians
   * @param lambda Rotation angle in radians
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param target Input target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, output_target_qubit)
   *
   * @par Example:
   * ```c++
   * {controls_out, target_out} = builder.mcu2(1.0, 0.5, {q0_in, q1_in}, q2_in);
   * ```
   * ```mlir
   * %controls_out, %target_out = flux.ctrl(%q0_in, %q1_in) %q2_in {
   *   %q2_res = flux.u2(1.0, 0.5) %q2_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q2_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit}) -> ({!flux.qubit,
   * !flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, Value> mcu2(const std::variant<double, Value>& phi,
                                    const std::variant<double, Value>& lambda,
                                    ValueRange controls, Value target);

  /**
   * @brief Apply a SWAP gate to two qubits
   *
   * @details
   * Consumes the input qubits and produces new output qubit SSA values.
   * The inputs are validated and the tracking is updated.
   *
   * @param qubit0 First input qubit (must be valid/unconsumed)
   * @param qubit1 Second input qubit (must be valid/unconsumed)
   * @return Output qubits
   *
   * @par Example:
   * ```c++
   * {q0_out, q1_out} = builder.swap(q0_in, q1_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out = flux.swap %q0_in, %q1_in : !flux.qubit, !flux.qubit ->
   * !flux.qubit, !flux.qubit
   * ```
   */
  std::pair<Value, Value> swap(Value qubit0, Value qubit1);

  /**
   * @brief Apply a controlled SWAP gate
   * @param control Input control qubit (must be valid/unconsumed)
   * @param qubit0 First target qubit (must be valid/unconsumed)
   * @param qubit1 Second target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubit, (output_qubit0, output_qubit1))
   *
   * @par Example:
   * ```c++
   * {q0_out, {q1_out, q2_out}} = builder.cswap(q0_in, q1_in, q2_in);
   * ```
   * ```mlir
   * %q0_out, %q1_out, %q2_out = flux.ctrl(%q0_in) %q1_in, %q2_in {
   *   %q1_res, %q2_res = flux.swap %q1_in, %q2_in : !flux.qubit,
   * !flux.qubit -> !flux.qubit, !flux.qubit
   *   flux.yield %q1_res, %q2_res
   * } : !flux.qubit, !flux.qubit, !flux.qubit -> !flux.qubit, !flux.qubit,
   * !flux.qubit
   * ```
   */
  std::pair<Value, std::pair<Value, Value>> cswap(Value control, Value qubit0,
                                                  Value qubit1);

  /**
   * @brief Apply a multi-controlled SWAP gate
   * @param controls Input control qubits (must be valid/unconsumed)
   * @param qubit0 First target qubit (must be valid/unconsumed)
   * @param qubit1 Second target qubit (must be valid/unconsumed)
   * @return Pair of (output_control_qubits, (output_qubit0, output_qubit1))
   *
   * @par Example:
   * ```c++
   * {controls_out, {q1_out, q2_out}} = builder.mcswap({q0_in, q1_in}, q2_in,
   * q3_in);
   * ```
   * ```mlir
   * %controls_out, %q2_out, %q3_out = flux.ctrl(%q0_in, %q1_in) %q2_in, %q3_in
   * { %q2_res, %q3_res = flux.swap %q2_in, %q3_in : !flux.qubit, !flux.qubit ->
   * !flux.qubit, !flux.qubit flux.yield %q2_res, %q3_res
   * } : ({!flux.qubit, !flux.qubit}, {!flux.qubit, !flux.qubit}) ->
   * ({!flux.qubit, !flux.qubit}, {!flux.qubit, !flux.qubit})
   * ```
   */
  std::pair<ValueRange, std::pair<Value, Value>>
  mcswap(ValueRange controls, Value qubit0, Value qubit1);

  //===--------------------------------------------------------------------===//
  // Modifiers
  //===--------------------------------------------------------------------===//

  /**
   * @brief Apply a controlled operation
   *
   * @param controls Control qubits
   * @param targets Target qubits
   * @param body Function that builds the body containing the target operation
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * controls_out, targets_out = builder.ctrl(q0_in, q1_in, [&](auto& b) {
   *   auto q1_res = b.x(q1_in);
   *   return {q1_res};
   * });
   * ```
   * ```mlir
   * %controls_out, %targets_out = flux.ctrl(%q0_in) %q1_in {
   *   %q1_res = flux.x %q1_in : !flux.qubit -> !flux.qubit
   *   flux.yield %q1_res
   * } : ({!flux.qubit}, {!flux.qubit}) -> ({!flux.qubit}, {!flux.qubit})
   * ```
   */
  std::pair<ValueRange, ValueRange>
  ctrl(ValueRange controls, ValueRange targets,
       const std::function<ValueRange(OpBuilder&, ValueRange)>& body);

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
   * flux.dealloc %q : !flux.qubit
   * ```
   */
  FluxProgramBuilder& dealloc(Value qubit);

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

  /**
   * @brief Helper to create a one-target, zero-parameter Flux operation
   *
   * @tparam OpType The operation type of the Flux operation
   * @param qubit Input qubit
   * @return Output qubit
   */
  template <typename OpType> Value createOneTargetZeroParameter(Value qubit);

  /**
   * @brief Helper to create a controlled one-target, zero-parameter Flux
   * operation
   *
   * @tparam OpType The operation type of the Flux operation
   * @param control Input control qubit
   * @param target Input target qubit
   * @return Pair of (output_control_qubit, output_target_qubit)
   */
  template <typename OpType>
  std::pair<Value, Value> createControlledOneTargetZeroParameter(Value control,
                                                                 Value target);

  /**
   * @brief Helper to create a multi-controlled one-target, zero-parameter Flux
   * operation
   *
   * @tparam OpType The operation type of the Flux operation
   * @param controls Input control qubits
   * @param target Input target qubit
   * @return Pair of (output_control_qubits, output_target_qubit)
   */
  template <typename OpType>
  std::pair<ValueRange, Value>
  createMultiControlledOneTargetZeroParameter(ValueRange controls,
                                              Value target);

  //===--------------------------------------------------------------------===//
  // Linear Type Tracking Helpers
  //===--------------------------------------------------------------------===//

  /**
   * @brief Validate that a qubit value is valid and unconsumed
   * @param qubit Qubit value to validate
   * @throws Aborts if qubit is not tracked (consumed or never created)
   */
  void validateQubitValue(Value qubit) const;

  /**
   * @brief Update tracking when an operation consumes and produces a qubit
   * @param inputQubit Input qubit being consumed (must be valid)
   * @param outputQubit New output qubit being produced
   */
  void updateQubitTracking(Value inputQubit, Value outputQubit);

  /// Track valid (unconsumed) qubit SSA values for linear type enforcement.
  /// Only values present in this set are valid for use in operations.
  /// When an operation consumes a qubit and produces a new one, the old value
  /// is removed and the new output is added.
  llvm::DenseSet<Value> validQubits;

  /// Track allocated classical Registers
  SmallVector<ClassicalRegister> allocatedClassicalRegisters;
};
} // namespace mlir::flux
