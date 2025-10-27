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
   * @param registerName The name of the classical register (e.g., "c")
   * @param registerIndex The index within the register for this measurement
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.measure(q0, "c", 0);
   * builder.measure(q1, "c", 1);
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
  QIRProgramBuilder& measure(Value qubit, StringRef registerName,
                             int64_t registerIndex);

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

  /**
   * @brief Apply the X gate to a qubit
   *
   * @param qubit Input qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.x(q);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr) -> ()
   * ```
   */
  QIRProgramBuilder& x(const Value qubit);

  /**
   * @brief Apply the RX gate to a qubit
   *
   * @param angle Rotation angle
   * @param qubit Input qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.rx(1.0, q);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__rx__body(%q, %c) : (!llvm.ptr, f64) -> ()
   * ```
   */
  QIRProgramBuilder& rx(double angle, const Value qubit);
  QIRProgramBuilder& rx(Value angle, const Value qubit);

  //===--------------------------------------------------------------------===//
  // Deallocation
  //===--------------------------------------------------------------------===//

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
  LLVM::LLVMFuncOp mainFunc;
  Location loc;

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

  /// Track allocated qubits for automatic deallocation
  DenseSet<Value> allocatedQubits;

  /// Cache static qubit pointers for reuse
  DenseMap<int64_t, Value> staticQubitCache;

  /// Cache result pointers for reuse (separate from qubits)
  DenseMap<int64_t, Value> resultPointerCache;

  /// Map from (register_name, register_index) to result pointer
  DenseMap<std::pair<StringRef, int64_t>, Value> registerResultMap;

  /// Track qubit and result counts for QIR metadata
  QIRMetadata metadata_;

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
};

} // namespace mlir::qir
