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
   * auto& c = builder.allocClassicalBitRegister(3, "c");
   * ```
   */
  ClassicalRegister& allocClassicalBitRegister(int64_t size,
                                               StringRef name = "c");

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

  /**
   * @brief Apply the X gate to a qubit
   *
   * @param qubit Input qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.x(qubit);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__x__body(%q) : (!llvm.ptr) -> ()
   * ```
   */
  QIRProgramBuilder& x(Value qubit);

  /**
   * @brief Apply the CX gate
   *
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cx(control, target);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__cx__body(%c, %t) : (!llvm.ptr, !llvm.ptr) -> ()
   * ```
   */
  QIRProgramBuilder& cx(Value control, Value target);

  /**
   * @brief Apply the RX gate to a qubit
   *
   * @param theta Rotation angle
   * @param qubit Input qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.rx(theta, qubit);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__rx__body(%q, %theta) : (!llvm.ptr, f64) -> ()
   * ```
   */
  QIRProgramBuilder& rx(const std::variant<double, Value>& theta, Value qubit);

  /**
   * @brief Apply the CRX
   *
   * @param theta Rotation angle
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.crx(theta, control, target);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__crx__body(%c, %t, %theta) : (!llvm.ptr,
   * !llvm.ptr, f64) -> ()
   * ```
   */
  QIRProgramBuilder& crx(const std::variant<double, Value>& theta,
                         Value control, Value target);

  /**
   * @brief Apply the U2 gate to a qubit
   *
   * @param phi Rotation angle
   * @param lambda Rotation angle
   * @param qubit Input qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.u2(phi, lambda, qubit);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__u2__body(%q, %phi, %lambda) : (!llvm.ptr, f64,
   * f64) -> ()
   * ```
   */
  QIRProgramBuilder& u2(const std::variant<double, Value>& phi,
                        const std::variant<double, Value>& lambda, Value qubit);

  /**
   * @brief Apply the CU2 gate
   *
   * @param phi Rotation angle
   * @param lambda Rotation angle
   * @param control Control qubit
   * @param target Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cu2(phi, lambda, control, target);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__cu2__body(%c, %t, %phi, %lambda) : (!llvm.ptr,
   * !llvm.ptr, f64, f64) -> ()
   * ```
   */
  QIRProgramBuilder& cu2(const std::variant<double, Value>& phi,
                         const std::variant<double, Value>& lambda,
                         Value control, Value target);

  /**
   * @brief Apply the SWAP gate to two qubits
   *
   * @param qubit0 Input qubit
   * @param qubit1 Input qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.swap(qubit0, qubit1);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__swap__body(%q0, %q1) : (!llvm.ptr, !llvm.ptr) ->
   * ()
   * ```
   */
  QIRProgramBuilder& swap(Value qubit0, Value qubit1);

  /**
   * @brief Apply the CSWAP gate
   *
   * @param control Control qubit
   * @param target1 Target qubit
   * @param target2 Target qubit
   * @return Reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * builder.cswap(control, target1, target2);
   * ```
   * ```mlir
   * llvm.call @__quantum__qis__cswap__body(%c, %t0, %t1) : (!llvm.ptr,
   * !llvm.ptr, !llvm.ptr) ->
   * ()
   * ```
   */
  QIRProgramBuilder& cswap(Value control, Value target0, Value target1);

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

  OpBuilder builder;
  ModuleOp module;
  Location loc;

private:
  /// Track allocated classical Registers
  SmallVector<ClassicalRegister> allocatedClassicalRegisters;

  LLVM::LLVMFuncOp mainFunc;

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
