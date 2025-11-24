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
      assert(0 <= index && index < size);
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

  /**
   * @brief Apply an Id gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.id(q);
   * ```
   * ```mlir
   * quartz.id %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& id(Value qubit);

  /**
   * @brief Apply a controlled Id gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cid(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.id %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cid(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Id gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcid({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.id %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcid(ValueRange controls, Value target);

  /**
   * @brief Apply an X gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.x(q);
   * ```
   * ```mlir
   * quartz.x %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& x(Value qubit);

  /**
   * @brief Apply a controlled X gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cx(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.x %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cx(Value control, Value target);

  /**
   * @brief Apply a multi-controlled X gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcx({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.x %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcx(ValueRange controls, Value target);

  /**
   * @brief Apply a Y gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.y(q);
   * ```
   * ```mlir
   * quartz.y %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& y(Value qubit);

  /**
   * @brief Apply a controlled Y gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cy(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.y %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cy(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Y gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcy({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.y %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcy(ValueRange controls, Value target);

  /**
   * @brief Apply a Z gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.z(q);
   * ```
   * ```mlir
   * quartz.z %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& z(Value qubit);

  /**
   * @brief Apply a controlled Z gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cz(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.z %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cz(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Z gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcz({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.z %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcz(ValueRange controls, Value target);

  /**
   * @brief Apply an H gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.h(q);
   * ```
   * ```mlir
   * quartz.h %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& h(Value qubit);

  /**
   * @brief Apply a controlled H gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.ch(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.h %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& ch(Value control, Value target);

  /**
   * @brief Apply a multi-controlled H gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mch({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.h %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mch(ValueRange controls, Value target);

  /**
   * @brief Apply an S gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.s(q);
   * ```
   * ```mlir
   * quartz.s %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& s(Value qubit);

  /**
   * @brief Apply a controlled S gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cs(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.s %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cs(Value control, Value target);

  /**
   * @brief Apply a multi-controlled S gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcs({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.s %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcs(ValueRange controls, Value target);

  /**
   * @brief Apply an Sdg gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.sdg(q);
   * ```
   * ```mlir
   * quartz.sdg %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& sdg(Value qubit);

  /**
   * @brief Apply a controlled Sdg gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.csdg(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.sdg %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& csdg(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Sdg gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcsdg({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.sdg %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcsdg(ValueRange controls, Value target);

  /**
   * @brief Apply a T gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.t(q);
   * ```
   * ```mlir
   * quartz.t %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& t(Value qubit);

  /**
   * @brief Apply a controlled T gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.ct(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.t %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& ct(Value control, Value target);

  /**
   * @brief Apply a multi-controlled T gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mct({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.t %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mct(ValueRange controls, Value target);

  /**
   * @brief Apply a Tdg gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.tdg(q);
   * ```
   * ```mlir
   * quartz.tdg %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& tdg(Value qubit);

  /**
   * @brief Apply a controlled Tdg gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.ctdg(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.tdg %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& ctdg(Value control, Value target);

  /**
   * @brief Apply a multi-controlled Tdg gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mctdg({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.tdg %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mctdg(ValueRange controls, Value target);

  /**
   * @brief Apply an SX gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.sx(q);
   * ```
   * ```mlir
   * quartz.sx %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& sx(Value qubit);

  /**
   * @brief Apply a controlled SX gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.csx(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.sx %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& csx(Value control, Value target);

  /**
   * @brief Apply a multi-controlled SX gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcsx({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.sx %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcsx(ValueRange controls, Value target);

  /**
   * @brief Apply an SXdg gate to a qubit
   *
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.sxdg(q);
   * ```
   * ```mlir
   * quartz.sxdg %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& sxdg(Value qubit);

  /**
   * @brief Apply a controlled SXdg gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.csxdg(q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.sxdg %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& csxdg(Value control, Value target);

  /**
   * @brief Apply a multi-controlled SXdg gate
   *
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcsxdg({q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.sxdg %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcsxdg(ValueRange controls, Value target);

  /**
   * @brief Apply an RX gate to a qubit
   *
   * @param theta Rotation angle in radians
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.rx(1.0, q);
   * ```
   * ```mlir
   * quartz.rx(%theta) %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& rx(const std::variant<double, Value>& theta,
                           Value qubit);

  /**
   * @brief Apply a CRX gate
   *
   * @param theta Rotation angle in radians
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.crx(1.0, q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.rx(%theta) %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& crx(const std::variant<double, Value>& theta,
                            Value control, Value target);

  /**
   * @brief Apply a multi-controlled RX gate
   *
   * @param theta Rotation angle in radians
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   * @par Example:
   * ```c++
   * builder.mcrx(1.0, {q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.rx(%theta) %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcrx(const std::variant<double, Value>& theta,
                             ValueRange controls, Value target);

  /**
   * @brief Apply an RY gate to a qubit
   *
   * @param theta Rotation angle in radians
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.ry(1.0, q);
   * ```
   * ```mlir
   * quartz.ry(%theta) %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& ry(const std::variant<double, Value>& theta,
                           Value qubit);

  /**
   * @brief Apply a CRY gate
   *
   * @param theta Rotation angle in radians
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cry(1.0, q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.ry(%theta) %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cry(const std::variant<double, Value>& theta,
                            Value control, Value target);

  /**
   * @brief Apply a multi-controlled RY gate
   *
   * @param theta Rotation angle in radians
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcry(1.0, {q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.ry(%theta) %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcry(const std::variant<double, Value>& theta,
                             ValueRange controls, Value target);

  /**
   * @brief Apply an RZ gate to a qubit
   *
   * @param theta Rotation angle in radians
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.rz(1.0, q);
   * ```
   * ```mlir
   * quartz.rz(%theta) %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& rz(const std::variant<double, Value>& theta,
                           Value qubit);

  /**
   * @brief Apply a CRZ gate
   *
   * @param theta Rotation angle in radians
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.crz(1.0, q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.rz(%theta) %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& crz(const std::variant<double, Value>& theta,
                            Value control, Value target);

  /**
   * @brief Apply a multi-controlled RZ gate
   *
   * @param theta Rotation angle in radians
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcrz(1.0, {q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.rz(%theta) %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcrz(const std::variant<double, Value>& theta,
                             ValueRange controls, Value target);

  /**
   * @brief Apply a P gate to a qubit
   *
   * @param theta Rotation angle in radians
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.p(1.0, q);
   * ```
   * ```mlir
   * quartz.p(%theta) %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& p(const std::variant<double, Value>& theta,
                          Value qubit);

  /**
   * @brief Apply a controlled P gate
   *
   * @param theta Rotation angle in radians
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cp(1.0, q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.p(%theta) %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cp(const std::variant<double, Value>& theta,
                           Value control, Value target);

  /**
   * @brief Apply a multi-controlled P gate
   *
   * @param theta Rotation angle in radians
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcp(1.0, {q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.p(%theta) %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcp(const std::variant<double, Value>& theta,
                            ValueRange controls, Value target);

  /**
   * @brief Apply an R gate to a qubit
   *
   * @param theta Rotation angle in radians
   * @param phi Rotation angle in radians
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.r(1.0, 0.5, q);
   * ```
   * ```mlir
   * quartz.r(%theta, %phi) %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& r(const std::variant<double, Value>& theta,
                          const std::variant<double, Value>& phi, Value qubit);

  /**
   * @brief Apply a controlled R gate
   *
   * @param theta Rotation angle in radians
   * @param phi Rotation angle in radians
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cr(1.0, 0.5, q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.r(%theta, %phi) %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cr(const std::variant<double, Value>& theta,
                           const std::variant<double, Value>& phi,
                           Value control, Value target);

  /**
   * @brief Apply a multi-controlled R gate
   *
   * @param theta Rotation angle in radians
   * @param phi Rotation angle in radians
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.mcr(1.0, 0.5, {q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.r(%theta, %phi) %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcr(const std::variant<double, Value>& theta,
                            const std::variant<double, Value>& phi,
                            ValueRange controls, Value target);

  /**
   * @brief Apply a U2 gate to a qubit
   *
   * @param phi Rotation angle in radians
   * @param lambda Rotation angle in radians
   * @param qubit Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.u2(1.0, 0.5, q);
   * ```
   * ```mlir
   * quartz.u2(%phi, %lambda) %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& u2(const std::variant<double, Value>& phi,
                           const std::variant<double, Value>& lambda,
                           Value qubit);

  /**
   * @brief Apply a controlled U2 gate
   *
   * @param phi Rotation angle in radians
   * @param lambda Rotation angle in radians
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   * @par Example:
   * ```c++
   * builder.cu2(1.0, 0.5, q0, q1);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.u2(%phi, %lambda) %q1 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cu2(const std::variant<double, Value>& phi,
                            const std::variant<double, Value>& lambda,
                            Value control, Value target);

  /**
   * @brief Apply a multi-controlled U2 gate
   *
   * @param phi Rotation angle in radians
   * @param lambda Rotation angle in radians
   * @param controls Control qubits
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   * @par Example:
   * ```c++
   * builder.mcu2(1.0, 0.5, {q0, q1}, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.u2(%phi, %lambda) %q2 : !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcu2(const std::variant<double, Value>& phi,
                             const std::variant<double, Value>& lambda,
                             ValueRange controls, Value target);

  /**
   * @brief Apply a SWAP gate to two qubits
   *
   * @param qubit0 First target qubit
   * @param qubit1 Second target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.swap(q0, q1);
   * ```
   * ```mlir
   * quartz.swap %q0, %q1 : !quartz.qubit, !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& swap(Value qubit0, Value qubit1);

  /**
   * @brief Apply a controlled SWAP gate
   *
   * @param control Control qubit
   * @param qubit0 First target qubit
   * @param qubit1 Second target qubit
   * @return Reference to this builder for method chaining
   * @par Example:
   * ```c++
   * builder.cswap(q0, q1, q2);
   * ```
   * ```mlir
   * quartz.ctrl(%q0) {
   *   quartz.swap %q1, %q2 : !quartz.qubit, !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& cswap(Value control, Value qubit0, Value qubit1);

  /**
   * @brief Apply a multi-controlled SWAP gate
   *
   * @param controls Control qubits
   * @param qubit0 First target qubit
   * @param qubit1 Second target qubit
   * @return Reference to this builder for method chaining
   * @par Example:
   * ```c++
   * builder.mcswap({q0, q1}, q2, q3);
   * ```
   * ```mlir
   * quartz.ctrl(%q0, %q1) {
   *   quartz.swap %q2, %q3 : !quartz.qubit, !quartz.qubit
   * }
   * ```
   */
  QuartzProgramBuilder& mcswap(ValueRange controls, Value qubit0, Value qubit1);

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
};
} // namespace mlir::quartz
