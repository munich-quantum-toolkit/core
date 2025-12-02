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

#include <cstddef>
#include <cstdint>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Operation.h>

namespace mlir::qir {

// QIR function names

static constexpr auto QIR_INITIALIZE = "__quantum__rt__initialize";
static constexpr auto QIR_MEASURE = "__quantum__qis__mz__body";
static constexpr auto QIR_RECORD_OUTPUT = "__quantum__rt__result_record_output";
static constexpr auto QIR_ARRAY_RECORD_OUTPUT =
    "__quantum__rt__array_record_output";
static constexpr auto QIR_RESET = "__quantum__qis__reset__body";

#define ADD_STANDARD_GATE(NAME_BIG, NAME_SMALL)                                \
  static constexpr auto QIR_##NAME_BIG =                                       \
      "__quantum__qis__" #NAME_SMALL "__body";                                 \
  static constexpr auto QIR_C##NAME_BIG =                                      \
      "__quantum__qis__c" #NAME_SMALL "__body";                                \
  static constexpr auto QIR_CC##NAME_BIG =                                     \
      "__quantum__qis__cc" #NAME_SMALL "__body";                               \
  static constexpr auto QIR_CCC##NAME_BIG =                                    \
      "__quantum__qis__ccc" #NAME_SMALL "__body";

ADD_STANDARD_GATE(GPHASE, gphase)
ADD_STANDARD_GATE(I, i)
ADD_STANDARD_GATE(X, x)
ADD_STANDARD_GATE(Y, y)
ADD_STANDARD_GATE(Z, z)
ADD_STANDARD_GATE(H, h)
ADD_STANDARD_GATE(S, s)
ADD_STANDARD_GATE(SDG, sdg)
ADD_STANDARD_GATE(T, t)
ADD_STANDARD_GATE(TDG, tdg)
ADD_STANDARD_GATE(SX, sx)
ADD_STANDARD_GATE(SXDG, sxdg)
ADD_STANDARD_GATE(RX, rx)
ADD_STANDARD_GATE(RY, ry)
ADD_STANDARD_GATE(RZ, rz)
ADD_STANDARD_GATE(P, p)
ADD_STANDARD_GATE(R, r)
ADD_STANDARD_GATE(U2, u2)
ADD_STANDARD_GATE(U, u3)
ADD_STANDARD_GATE(SWAP, swap)
ADD_STANDARD_GATE(ISWAP, iswap)
ADD_STANDARD_GATE(DCX, dcx)
ADD_STANDARD_GATE(ECR, ecr)
ADD_STANDARD_GATE(RXX, rxx)
ADD_STANDARD_GATE(RYY, ryy)
ADD_STANDARD_GATE(RZX, rzx)
ADD_STANDARD_GATE(RZZ, rzz)
ADD_STANDARD_GATE(XXPLUSYY, xx_plus_yy)
ADD_STANDARD_GATE(XXMINUSYY, xx_minus_yy)

#undef ADD_STANDARD_GATE

// Functions for getting QIR function names

#define DEFINE_GETTER(NAME)                                                    \
  /**                                                                          \
   * @brief Gets the QIR function name for NAME                                \
   *                                                                           \
   * @param numControls Number of control qubits                               \
   * @return The QIR function name                                             \
   */                                                                          \
  inline StringRef getFnName##NAME(size_t numControls) {                       \
    switch (numControls) {                                                     \
    case 0:                                                                    \
      return QIR_##NAME;                                                       \
    case 1:                                                                    \
      return QIR_C##NAME;                                                      \
    case 2:                                                                    \
      return QIR_CC##NAME;                                                     \
    case 3:                                                                    \
      return QIR_CCC##NAME;                                                    \
    default:                                                                   \
      llvm::report_fatal_error(                                                \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
  }

DEFINE_GETTER(GPHASE)
DEFINE_GETTER(I)
DEFINE_GETTER(X)
DEFINE_GETTER(Y)
DEFINE_GETTER(Z)
DEFINE_GETTER(H)
DEFINE_GETTER(S)
DEFINE_GETTER(SDG)
DEFINE_GETTER(T)
DEFINE_GETTER(TDG)
DEFINE_GETTER(SX)
DEFINE_GETTER(SXDG)
DEFINE_GETTER(RX)
DEFINE_GETTER(RY)
DEFINE_GETTER(RZ)
DEFINE_GETTER(P)
DEFINE_GETTER(R)
DEFINE_GETTER(U2)
DEFINE_GETTER(U)
DEFINE_GETTER(SWAP)
DEFINE_GETTER(ISWAP)
DEFINE_GETTER(DCX)
DEFINE_GETTER(ECR)
DEFINE_GETTER(RXX)
DEFINE_GETTER(RYY)
DEFINE_GETTER(RZX)
DEFINE_GETTER(RZZ)
DEFINE_GETTER(XXPLUSYY)
DEFINE_GETTER(XXMINUSYY)

#undef DEFINE_GETTER

/**
 * @brief State object for tracking QIR metadata during conversion
 *
 * @details
 * This struct maintains metadata about the QIR program being built:
 * - Qubit and result counts for QIR metadata
 * - Whether dynamic memory management is needed
 */
struct QIRMetadata {
  /// Number of qubits used in the module
  size_t numQubits{0};
  /// Number of measurement results stored in the module
  size_t numResults{0};
  /// Whether the module uses dynamic qubit management
  bool useDynamicQubit{false};
  /// Whether the module uses dynamic result management
  bool useDynamicResult{false};
};

/**
 * @brief Find the main LLVM function with entry_point attribute
 *
 * @details
 * Searches for the LLVM function marked with the "entry_point" attribute in
 * the passthrough attributes.
 *
 * @param op The module operation to search in
 * @return The main LLVM function, or nullptr if not found
 */
LLVM::LLVMFuncOp getMainFunction(Operation* op);

/**
 * @brief Set QIR base profile metadata attributes on the main function
 *
 * @details
 * Adds the required metadata attributes for QIR base profile compliance:
 * - `entry_point`: Marks the main entry point function
 * - `output_labeling_schema`: schema_id
 * - `qir_profiles`: base_profile
 * - `required_num_qubits`: Number of qubits used
 * - `required_num_results`: Number of measurement results
 * - `qir_major_version`: 1
 * - `qir_minor_version`: 0
 * - `dynamic_qubit_management`: true/false
 * - `dynamic_result_management`: true/false
 *
 * These attributes are required by the QIR specification and inform QIR
 * consumers about the module's resource requirements and capabilities.
 *
 * @param main The main LLVM function to annotate
 * @param metadata The QIR metadata containing qubit/result counts
 */
void setQIRAttributes(LLVM::LLVMFuncOp& main, const QIRMetadata& metadata);

/**
 * @brief Get or create a QIR function declaration
 *
 * @details
 * Searches for an existing function declaration in the symbol table. If not
 * found, creates a new function declaration at the end of the module.
 *
 * For QIR functions that are irreversible (measurement, reset, deallocation),
 * the "irreversible" attribute is added automatically.
 *
 * @param builder The builder to use for creating operations
 * @param op The operation requesting the function (for context)
 * @param fnName The name of the QIR function
 * @param fnType The LLVM function type signature
 * @return The LLVM function declaration
 */
LLVM::LLVMFuncOp getOrCreateFunctionDeclaration(OpBuilder& builder,
                                                Operation* op, StringRef fnName,
                                                Type fnType);

/**
 * @brief Create a global string constant for result labeling
 *
 * @details
 * Creates a global string constant at module level and returns an addressOf
 * operation pointing to it. The global is created at the start of the module,
 * and the addressOf is inserted at the builder's current insertion point.
 *
 * @param builder The builder to use for creating operations
 * @param op The operation requesting the label (for context/location)
 * @param label The label string (e.g., "r0")
 * @param symbolPrefix The prefix for the symbol name (default:
 * "qir.result_label")
 * @return AddressOf operation for the global constant
 */
LLVM::AddressOfOp
createResultLabel(OpBuilder& builder, Operation* op, StringRef label,
                  StringRef symbolPrefix = "qir.result_label");

/**
 * @brief Create a pointer value from an integer index
 *
 * @details
 * Creates a constant operation with the given index and converts it to a
 * pointer using inttoptr. This is used for static qubit/result references in
 * QIR.
 *
 * @param builder The builder to use for creating operations
 * @param loc The location for the operations
 * @param index The integer index
 * @return The pointer value
 */
Value createPointerFromIndex(OpBuilder& builder, Location loc, int64_t index);

} // namespace mlir::qir
