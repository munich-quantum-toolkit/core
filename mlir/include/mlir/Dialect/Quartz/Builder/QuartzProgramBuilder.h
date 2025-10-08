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
 * The QuartzProgramBuilder provides a type-safe, ergonomic interface for
 * programmatically constructing quantum circuits using reference semantics.
 * Gates operate on qubit references without explicit SSA value threading,
 * making it natural to express imperative quantum programs similar to how
 * hardware physically transforms quantum states.
 *
 * The builder follows the Quartz dialect's philosophy of reference semantics,
 * where operations modify qubits in place. This provides:
 * - Natural mapping to hardware execution models
 * - Intuitive representation for circuit descriptions
 * - Direct compatibility with imperative quantum programming languages
 * - Support for method chaining through fluent interface design
 *
 * @par Example Usage:
 * ```c++
 * QuartzProgramBuilder builder(context);
 * builder.initialize();
 *
 * auto q0 = builder.staticQubit(0);
 * auto q1 = builder.staticQubit(1);
 *
 * // Create Bell state using method chaining
 * builder.h(q0).cx(q0, q1);
 *
 * auto module = builder.finalize();
 * ```
 *
 * This produces the following MLIR module with a main function:
 * ```mlir
 * module {
 *   func.func @main() attributes {passthrough = ["entry_point"]} {
 *     %q0 = quartz.static 0 : !quartz.qubit
 *     %q1 = quartz.static 1 : !quartz.qubit
 *     quartz.h %q0 : !quartz.qubit
 *     quartz.cx %q0, %q1 : !quartz.qubit, !quartz.qubit
 *     func.return
 *   }
 * }
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
   * This method must be called before any operations are added to the program.
   * It creates a main function with an entry_point attribute and sets up the
   * builder's insertion point. All subsequent operations will be added to the
   * body of this main function.
   *
   * The generated function structure:
   * ```mlir
   * func.func @main() attributes {passthrough = ["entry_point"]} {
   *   // Operations added here
   * }
   * ```
   */
  void initialize();

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /**
   * @brief Dynamically allocate a single qubit
   *
   * @details
   * Allocates a new qubit dynamically and returns a reference to it.
   * The qubit is initialized to the |0⟩ state.
   *
   * @return A Value representing the allocated qubit
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubit();
   * ```
   *
   * This generates:
   * ```mlir
   * %q = quartz.alloc : !quartz.qubit
   * ```
   */
  Value allocQubit();

  /**
   * @brief Get a static reference to a qubit by index
   *
   * @details
   * Creates a reference to a qubit identified by a static index. This is
   * useful for referring to fixed qubits in a quantum program or to
   * hardware-mapped qubits.
   *
   * @param index The index of the qubit to reference
   * @return A Value representing the qubit at the given index
   *
   * @par Example:
   * ```c++
   * auto q0 = builder.staticQubit(0);
   * auto q1 = builder.staticQubit(1);
   * ```
   *
   * This generates:
   * ```mlir
   * %q0 = quartz.static 0 : !quartz.qubit
   * %q1 = quartz.static 1 : !quartz.qubit
   * ```
   */
  Value staticQubit(size_t index);

  /**
   * @brief Allocate a qubit register as a sequence of individual qubits
   *
   * @details
   * Allocates multiple qubits that form a logical register. Each qubit is
   * allocated individually with metadata indicating its membership in the
   * register, including the register name, size, and index within the register.
   *
   * The returned SmallVector can be used directly or converted to ValueRange
   * as needed.
   *
   * @param size The number of qubits in the register
   * @param name The symbolic name for the register (default: "q")
   * @return A SmallVector containing the allocated qubits
   *
   * @par Example:
   * ```c++
   * auto q = builder.allocQubitRegister(3, "q");
   * // Use q[0], q[1], q[2] directly
   * ```
   *
   * This generates:
   * ```mlir
   * %q0 = quartz.alloc q[3, 0] : !quartz.qubit
   * %q1 = quartz.alloc q[3, 1] : !quartz.qubit
   * %q2 = quartz.alloc q[3, 2] : !quartz.qubit
   * ```
   */
  SmallVector<Value> allocQubitRegister(size_t size, StringRef name = "q");

  /**
   * @brief Allocate a classical bit register
   *
   * @details
   * Allocates a classical register for storing measurement results or other
   * classical data. The register is represented as a memref of i1 elements.
   *
   * @param size The number of bits in the register
   * @param name The symbolic name for the register (default: "c")
   * @return A Value representing the allocated memref of i1 elements
   *
   * @par Example:
   * ```c++
   * auto c = builder.allocClassicalBitRegister(3, "c");
   * ```
   *
   * This generates:
   * ```mlir
   * %c = memref.alloc() {sym_name = "c"} : memref<3xi1>
   * ```
   */
  Value allocClassicalBitRegister(size_t size, StringRef name = "c");

  //===--------------------------------------------------------------------===//
  // Measurement and Reset
  //===--------------------------------------------------------------------===//

  /**
   * @brief Measure a qubit and return the measurement result
   *
   * @details
   * Measures a qubit in the computational (Z) basis, collapsing the state
   * and returning a classical bit result (i1).
   *
   * @param qubit The qubit to measure
   * @return A Value representing the classical measurement result (i1)
   *
   * @par Example:
   * ```c++
   * auto q = builder.staticQubit(0);
   * auto result = builder.measure(q);
   * ```
   *
   * This generates:
   * ```mlir
   * %q = quartz.static 0 : !quartz.qubit
   * %result = quartz.measure %q : !quartz.qubit -> i1
   * ```
   */
  Value measure(Value qubit);

  /**
   * @brief Measure a qubit and store the result in a classical register
   *
   * @details
   * Measures a qubit in the computational basis and stores the result at
   * the specified index in a classical register (memref). This is useful
   * for accumulating multiple measurement results in a register.
   *
   * @param qubit The qubit to measure
   * @param memref The classical register to store the result in
   * @param index The index in the register where the result should be stored
   * @return A reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * auto q0 = builder.staticQubit(0);
   * auto q1 = builder.staticQubit(1);
   * auto c = builder.allocClassicalBitRegister(2);
   *
   * builder.measure(q0, c, 0)
   *        .measure(q1, c, 1);
   * ```
   *
   * This generates (within the main function body):
   * ```mlir
   * %q0 = quartz.static 0 : !quartz.qubit
   * %q1 = quartz.static 1 : !quartz.qubit
   * %c = memref.alloc() {sym_name = "c"} : memref<2xi1>
   * %r0 = quartz.measure %q0 : !quartz.qubit -> i1
   * %c0 = arith.constant 0 : index
   * memref.store %r0, %c[%c0] : memref<2xi1>
   * %r1 = quartz.measure %q1 : !quartz.qubit -> i1
   * %c1 = arith.constant 1 : index
   * memref.store %r1, %c[%c1] : memref<2xi1>
   * ```
   */
  QuartzProgramBuilder& measure(Value qubit, Value memref, size_t index);

  /**
   * @brief Reset a qubit to the |0⟩ state
   *
   * @details
   * Resets a qubit to the |0⟩ state, regardless of its current state.
   * This operation is often used to recycle qubits in quantum algorithms
   * or to initialize qubits to a known state.
   *
   * @param qubit The qubit to reset
   * @return A reference to this builder for method chaining
   *
   * @par Example:
   * ```c++
   * auto q = builder.staticQubit(0);
   * builder.reset(q);
   * ```
   *
   * This generates:
   * ```mlir
   * %q = quartz.static 0 : !quartz.qubit
   * quartz.reset %q : !quartz.qubit
   * ```
   */
  QuartzProgramBuilder& reset(Value qubit);

  //===--------------------------------------------------------------------===//
  // Finalization
  //===--------------------------------------------------------------------===//

  /**
   * @brief Finalize the program and return the constructed module
   *
   * @details
   * Completes the construction of the quantum program by:
   * 1. Adding a return statement to the main function
   * 2. Transferring ownership of the module to the caller
   *
   * After calling this method, the builder is invalidated and should not be
   * used to add more operations. The returned OwningOpRef takes ownership of
   * the module.
   *
   * @return OwningOpRef containing the constructed quantum program module
   *
   * @par Example:
   * ```c++
   * QuartzProgramBuilder builder(context);
   * builder.initialize();
   * auto q = builder.staticQubit(0);
   * builder.h(q);
   * auto module = builder.finalize();
   * // module now owns the MLIR module, builder is invalidated
   * ```
   */
  OwningOpRef<ModuleOp> finalize();

private:
  OpBuilder builder;
  ModuleOp module;
  Location loc;
};
} // namespace mlir::quartz
