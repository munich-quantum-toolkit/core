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

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

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
 * @par QIR Structure:
 * QIR programs follow a specific structure with:
 * - Entry block: Constants and initialization (__quantum__rt__initialize)
 * - Main block: Reversible quantum operations (gates)
 * - Irreversible block: Measurements, resets, deallocations
 * - End block: Return statement
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
 * auto result = builder.measure(q0, 0);
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
   * @brief Allocate a single qubit dynamically
   * @return An LLVM pointer representing the qubit
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubit();
   * ```
   * ```mlir
   * %q = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
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
   * @brief Allocate a qubit register dynamically
   * @param size Number of qubits (must be positive)
   * @return Vector of LLVM pointers representing the qubits
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubitRegister(3);
   * ```
   * ```mlir
   * %q0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
   * %q1 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
   * %q2 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
   * ```
   */
  SmallVector<Value> allocQubitRegister(int64_t size);

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measure a qubit and record the result
   *
   * @details
   * Performs a Z-basis measurement using __quantum__qis__mz__body and
   * records the result using __quantum__rt__result_record_output. The
   * result is labeled with the provided index (e.g., "r0", "r1").
   *
   * @param qubit The qubit to measure
   * @param resultIndex The classical bit index for result labeling
   * @return An LLVM pointer to the measurement result
   *
   * @par Example:
   * ```c++
   * auto result = builder.measure(q0, 0);
   * ```
   * ```mlir
   * %c0 = llvm.mlir.constant(0 : i64) : i64
   * %r = llvm.inttoptr %c0 : i64 to !llvm.ptr
   * llvm.call @__quantum__qis__mz__body(%q0, %r) : (!llvm.ptr, !llvm.ptr) ->
   * () llvm.call @__quantum__rt__result_record_output(%r, %label0) :
   * (!llvm.ptr, !llvm.ptr) -> ()
   * ```
   */
  Value measure(Value qubit, size_t resultIndex);

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

  /**
   * @brief Explicitly deallocate a qubit
   *
   * @details
   * Deallocates a qubit using __quantum__rt__qubit_release. Optional -
   * finalize() automatically deallocates all remaining allocated qubits.
   *
   * @param qubit The qubit to deallocate
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.dealloc(q);
   * ```
   * ```mlir
   * llvm.call @__quantum__rt__qubit_release(%q) : (!llvm.ptr) -> ()
   * ```
   */
  QIRProgramBuilder& dealloc(Value qubit);

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Finalize the program and return the constructed module
   *
   * @details
   * Automatically deallocates all remaining allocated qubits, ensures proper
   * QIR metadata attributes are set, and transfers ownership of the module to
   * the caller. The builder should not be used after calling this method.
   *
   * @return OwningOpRef containing the constructed QIR program module
   */
  OwningOpRef<ModuleOp> finalize();

private:
  OpBuilder builder;
  ModuleOp module;
  LLVM::LLVMFuncOp mainFunc;
  Location loc;

  /// Entry block: constants and initialization
  Block* entryBlock{};
  /// Main block: reversible operations (gates)
  Block* mainBlock{};
  /// Irreversible block: measurements, resets, deallocations
  Block* irreversibleBlock{};
  /// End block: return statement
  Block* endBlock{};

  /// Track allocated qubits for automatic deallocation
  llvm::DenseSet<Value> allocatedQubits;

  /// Cache static qubit pointers for reuse
  llvm::DenseMap<size_t, Value> staticQubitCache;

  /// Cache result pointers for reuse (separate from qubits)
  llvm::DenseMap<size_t, Value> resultPointerCache;

  /// Track result labels to avoid duplicates
  llvm::DenseMap<size_t, LLVM::AddressOfOp> resultLabelCache;

  /// Track qubit and result counts for QIR metadata
  QIRMetadata metadata_;
};

} // namespace mlir::qir
