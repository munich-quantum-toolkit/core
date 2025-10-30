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
class FluxProgramBuilder {
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
   * @brief Apply the X gate to a qubit
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
   * @brief Apply the RX gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param theta Rotation angle
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
  Value rx(std::variant<double, Value> theta, Value qubit);

  /**
   * @brief Apply the U2 gate to a qubit
   *
   * @details
   * Consumes the input qubit and produces a new output qubit SSA value.
   * The input is validated and the tracking is updated.
   *
   * @param phi Rotation angle
   * @param lambda Rotation angle
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
  Value u2(std::variant<double, Value> phi, std::variant<double, Value> lambda,
           Value qubit);

  /**
   * @brief Apply the SWAP gate to two qubits
   *
   * @details
   * Consumes the input qubits and produces new output qubit SSA values.
   * The inputs are validated and the tracking is updated.
   *
   * @param qubit0 Input qubit (must be valid/unconsumed)
   * @param qubit1 Input qubit (must be valid/unconsumed)
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

  OpBuilder builder;
  ModuleOp module;
  Location loc;

private:
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
