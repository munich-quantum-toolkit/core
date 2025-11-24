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

// QIR function name constants
static constexpr auto QIR_INITIALIZE = "__quantum__rt__initialize";
static constexpr auto QIR_MEASURE = "__quantum__qis__mz__body";
static constexpr auto QIR_RECORD_OUTPUT = "__quantum__rt__result_record_output";
static constexpr auto QIR_ARRAY_RECORD_OUTPUT =
    "__quantum__rt__array_record_output";
static constexpr auto QIR_RESET = "__quantum__qis__reset__body";
static constexpr auto QIR_ID = "__quantum__qis__i__body";
static constexpr auto QIR_CID = "__quantum__qis__ci__body";
static constexpr auto QIR_CCID = "__quantum__qis__cci__body";
static constexpr auto QIR_CCCID = "__quantum__qis__ccci__body";
static constexpr auto QIR_X = "__quantum__qis__x__body";
static constexpr auto QIR_CX = "__quantum__qis__cx__body";
static constexpr auto QIR_CCX = "__quantum__qis__ccx__body";
static constexpr auto QIR_CCCX = "__quantum__qis__cccx__body";
static constexpr auto QIR_Y = "__quantum__qis__y__body";
static constexpr auto QIR_CY = "__quantum__qis__cy__body";
static constexpr auto QIR_CCY = "__quantum__qis__ccy__body";
static constexpr auto QIR_CCCY = "__quantum__qis__cccy__body";
static constexpr auto QIR_Z = "__quantum__qis__z__body";
static constexpr auto QIR_CZ = "__quantum__qis__cz__body";
static constexpr auto QIR_CCZ = "__quantum__qis__ccz__body";
static constexpr auto QIR_CCCZ = "__quantum__qis__cccz__body";
static constexpr auto QIR_H = "__quantum__qis__h__body";
static constexpr auto QIR_CH = "__quantum__qis__ch__body";
static constexpr auto QIR_CCH = "__quantum__qis__cch__body";
static constexpr auto QIR_CCCH = "__quantum__qis__ccch__body";
static constexpr auto QIR_S = "__quantum__qis__s__body";
static constexpr auto QIR_CS = "__quantum__qis__cs__body";
static constexpr auto QIR_CCS = "__quantum__qis__ccs__body";
static constexpr auto QIR_CCCS = "__quantum__qis__cccs__body";
static constexpr auto QIR_SDG = "__quantum__qis__sdg__body";
static constexpr auto QIR_CSDG = "__quantum__qis__csdg__body";
static constexpr auto QIR_CCSDG = "__quantum__qis__ccsdg__body";
static constexpr auto QIR_CCCSDG = "__quantum__qis__cccsdg__body";
static constexpr auto QIR_T = "__quantum__qis__t__body";
static constexpr auto QIR_CT = "__quantum__qis__ct__body";
static constexpr auto QIR_CCT = "__quantum__qis__cct__body";
static constexpr auto QIR_CCCT = "__quantum__qis__ccct__body";
static constexpr auto QIR_TDG = "__quantum__qis__tdg__body";
static constexpr auto QIR_CTDG = "__quantum__qis__ctdg__body";
static constexpr auto QIR_CCTDG = "__quantum__qis__cctdg__body";
static constexpr auto QIR_CCCTDG = "__quantum__qis__ccctdg__body";
static constexpr auto QIR_SX = "__quantum__qis__sx__body";
static constexpr auto QIR_CSX = "__quantum__qis__csx__body";
static constexpr auto QIR_CCSX = "__quantum__qis__ccsx__body";
static constexpr auto QIR_CCCSX = "__quantum__qis__cccsx__body";
static constexpr auto QIR_SXDG = "__quantum__qis__sxdg__body";
static constexpr auto QIR_CSXDG = "__quantum__qis__csxdg__body";
static constexpr auto QIR_CCSXDG = "__quantum__qis__ccsxdg__body";
static constexpr auto QIR_CCCSXDG = "__quantum__qis__cccsxdg__body";
static constexpr auto QIR_RX = "__quantum__qis__rx__body";
static constexpr auto QIR_CRX = "__quantum__qis__crx__body";
static constexpr auto QIR_CCRX = "__quantum__qis__ccrx__body";
static constexpr auto QIR_CCCRX = "__quantum__qis__cccrx__body";
static constexpr auto QIR_RY = "__quantum__qis__ry__body";
static constexpr auto QIR_CRY = "__quantum__qis__cry__body";
static constexpr auto QIR_CCRY = "__quantum__qis__ccry__body";
static constexpr auto QIR_CCCRY = "__quantum__qis__cccry__body";
static constexpr auto QIR_RZ = "__quantum__qis__rz__body";
static constexpr auto QIR_CRZ = "__quantum__qis__crz__body";
static constexpr auto QIR_CCRZ = "__quantum__qis__ccrz__body";
static constexpr auto QIR_CCCRZ = "__quantum__qis__cccrz__body";
static constexpr auto QIR_P = "__quantum__qis__p__body";
static constexpr auto QIR_CP = "__quantum__qis__cp__body";
static constexpr auto QIR_CCP = "__quantum__qis__ccp__body";
static constexpr auto QIR_CCCP = "__quantum__qis__cccp__body";
static constexpr auto QIR_U2 = "__quantum__qis__u2__body";
static constexpr auto QIR_CU2 = "__quantum__qis__cu2__body";
static constexpr auto QIR_CCU2 = "__quantum__qis__ccu2__body";
static constexpr auto QIR_CCCU2 = "__quantum__qis__cccu2__body";
static constexpr auto QIR_SWAP = "__quantum__qis__swap__body";
static constexpr auto QIR_CSWAP = "__quantum__qis__cswap__body";
static constexpr auto QIR_CCSWAP = "__quantum__qis__ccswap__body";
static constexpr auto QIR_CCCSWAP = "__quantum__qis__cccswap__body";

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
