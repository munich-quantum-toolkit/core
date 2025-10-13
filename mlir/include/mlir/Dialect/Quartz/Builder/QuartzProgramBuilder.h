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

#include <cstddef>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

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
class QuartzProgramBuilder {
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
   * @brief Allocate a classical bit register
   * @param size Number of bits
   * @param name Register name (default: "c")
   * @return Memref of i1 elements
   *
   * @par Example:
   * ```c++
   * auto c = builder.allocClassicalBitRegister(3, "c");
   * ```
   * ```mlir
   * %c = memref.alloca() {sym_name = "c"} : memref<3xi1>
   * ```
   */
  Value allocClassicalBitRegister(int64_t size, StringRef name = "c");

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
   * @brief Measure a qubit and store the result in a classical register
   *
   * @param qubit The qubit to measure
   * @param memref Classical register
   * @param index Storage index
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.measure(q0, c, 0);
   * ```
   * ```mlir
   * %r0 = quartz.measure %q0 : !quartz.qubit -> i1
   * %c0 = arith.constant 0 : index
   * memref.store %r0, %c[%c0] : memref<2xi1>
   * ```
   */
  QuartzProgramBuilder& measure(Value qubit, Value memref, int64_t index);

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
   * statement, and transfers ownership of the module to the caller.
   * The builder should not be used after calling this method.
   *
   * @return OwningOpRef containing the constructed quantum program module
   */
  OwningOpRef<ModuleOp> finalize();

private:
  OpBuilder builder;
  ModuleOp module;
  Location loc;

  /// Track allocated qubits for automatic deallocation
  llvm::DenseSet<Value> allocatedQubits;
};
} // namespace mlir::quartz
