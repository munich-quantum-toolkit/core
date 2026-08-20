/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace llvm {
class Module;
} // namespace llvm

namespace mlir {
class OpBuilder;
class Operation;
namespace LLVM {
class AddressOfOp;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace mlir::qir {

/// Normalize QIR profile module flags after MLIR-to-LLVM translation.
///
/// MLIR 22 translates integer-valued `llvm.module_flags` attributes to i32
/// metadata and only supports array-valued flags for LLVM's own CG profile.
/// QIR instead requires i1/i2 capability flags and metadata tuples describing
/// the integer and floating-point widths used by Adaptive Profile classical
/// computations. This function repairs the scalar flag widths and derives the
/// optional Adaptive Profile flags from the translated LLVM module.
void normalizeQIRModuleFlags(llvm::Module& moduleOp, bool useAdaptive);

// QIR function names

inline constexpr auto QIR_QUBIT_ARRAY_ALLOC =
    "__quantum__rt__qubit_array_allocate";
inline constexpr auto QIR_QUBIT_ARRAY_RELEASE =
    "__quantum__rt__qubit_array_release";

inline constexpr auto QIR_QUBIT_ALLOC = "__quantum__rt__qubit_allocate";
inline constexpr auto QIR_QUBIT_RELEASE = "__quantum__rt__qubit_release";

inline constexpr auto QIR_RESULT_ARRAY_ALLOC =
    "__quantum__rt__result_array_allocate";
inline constexpr auto QIR_RESULT_ARRAY_RECORD_OUTPUT =
    "__quantum__rt__result_array_record_output";
inline constexpr auto QIR_RESULT_ARRAY_RELEASE =
    "__quantum__rt__result_array_release";

inline constexpr auto QIR_RESULT_ALLOC = "__quantum__rt__result_allocate";
inline constexpr auto QIR_RESULT_RELEASE = "__quantum__rt__result_release";

inline constexpr auto QIR_INITIALIZE = "__quantum__rt__initialize";
inline constexpr auto QIR_MEASURE = "__quantum__qis__mz__body";
inline constexpr auto QIR_READ_RESULT = "__quantum__rt__read_result";
inline constexpr auto QIR_RECORD_OUTPUT = "__quantum__rt__result_record_output";
inline constexpr auto QIR_ARRAY_RECORD_OUTPUT =
    "__quantum__rt__array_record_output";
inline constexpr auto QIR_RESET = "__quantum__qis__reset__body";

inline constexpr auto QIR_GPHASE = "__quantum__qis__gphase__body";

inline constexpr auto QIR_ARRAY_CREATE = "__quantum__rt__array_create_1d";
inline constexpr auto QIR_ARRAY_ELEMENT =
    "__quantum__rt__array_get_element_ptr_1d";
inline constexpr auto QIR_ARRAY_RELEASE =
    "__quantum__rt__array_update_reference_count";
inline constexpr auto QIR_TUPLE_CREATE = "__quantum__rt__tuple_create";
inline constexpr auto QIR_TUPLE_RELEASE =
    "__quantum__rt__tuple_update_reference_count";

#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  inline constexpr auto QIR_##GETTER = "__quantum__qis__" #NAME "__" #SUFFIX;  \
  inline constexpr auto QIR_C##GETTER =                                        \
      "__quantum__qis__c" #NAME "__" #SUFFIX;                                  \
  inline constexpr auto QIR_CC##GETTER =                                       \
      "__quantum__qis__cc" #NAME "__" #SUFFIX;                                 \
  inline constexpr auto QIR_##GETTER##_CTL =                                   \
      "__quantum__qis__" #NAME "__" #CTL_SUFFIX;
#include "mlir/Conversion/GateTable.def"

inline StringRef selectQISFunctionName(const StringRef body,
                                       const StringRef singleControlled,
                                       const StringRef doubleControlled,
                                       const StringRef genericControlled,
                                       const size_t numControls) {
  switch (numControls) {
  case 0:
    return body;
  case 1:
    return singleControlled;
  case 2:
    return doubleControlled;
  default:
    return genericControlled;
  }
}

#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  inline StringRef getFnName##GETTER(const size_t numControls) {               \
    return selectQISFunctionName(QIR_##GETTER, QIR_C##GETTER, QIR_CC##GETTER,  \
                                 QIR_##GETTER##_CTL, numControls);             \
  }
#include "mlir/Conversion/GateTable.def"

/**
 * @brief Emit a QIS call, materializing generic controlled arguments when
 * required.
 *
 * Body, adjoint, and one- or two-control calls pass their arguments directly.
 * Calls with three or more controls use the generic controlled specialization:
 * the first argument is an array of controls and the second is either the
 * single target or a tuple containing parameters followed by targets.
 *
 */
void emitQISCall(OpBuilder& builder, Operation* anchor, Location loc,
                 ValueRange parameters, ValueRange controls, ValueRange targets,
                 StringRef fnName);

/**
 * @brief Find the main LLVM function
 *
 * @details
 * Searches first for the MQT program entry-point marker. It also accepts the
 * lowered QIR `entry_point` passthrough attribute.
 *
 * @param op The module operation to search in
 * @return The main LLVM function, or nullptr if not found
 */
LLVM::LLVMFuncOp getMainFunction(Operation* op);

/**
 * @brief Get or create a QIR function declaration
 *
 * @details
 * Searches for an existing function declaration in the symbol table. If not
 * found, creates a new function declaration at the end of the module.
 *
 * For QIR functions that are irreversible (measurement and reset), the
 * "irreversible" attribute is added automatically.
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
 * Creates a global string constant at the module level and inserts an
 * AddressOfOp at the start of the main function's entry block.
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

/// A classical bit register.
struct ClassicalRegister {
  /// Label of the register (e.g., "c0").
  std::string label;
  /// Whether the register should be recorded in the output.
  bool record = true;
  /// Number of bits in the register.
  std::variant<int64_t, Value> size = int64_t{0};
  /// Base Profile: Pre-allocated result pointer for each bit.
  SmallVector<Value> results;
  /// Adaptive Profile: The backing result array.
  Value array;
};

/// A static result (i.e., a result that is not part of a classical register).
struct StaticResult {
  /// The result pointer.
  Value pointer;
  /// Whether the result should be recorded in the output.
  bool record = false;
};

/**
 * @brief Emit the output-recording calls.
 *
 * @param builder The builder to use
 * @param anchor An operation used to locate the enclosing module
 * @param classicalRegisters The classical registers to record. If `record` is
 * not set, the register is skipped.
 * @param staticResults The static results to record. If `record` is not set,
 * the result is skipped.
 */
void emitOutputRecording(OpBuilder& builder, Operation* anchor,
                         ArrayRef<ClassicalRegister> classicalRegisters,
                         const DenseMap<int64_t, StaticResult>& staticResults);

/**
 * @brief Helper to resolve a variant of either `int64_t` type or `Value` type
 * to a `Value`
 *
 * @details
 * Helper function to resolve a given variant to a `Value`. Creates an
 * `LLVM::ConstantOp` from the `int64_t` value. If the variant holds a `Value`,
 * return it directly.
 */
[[nodiscard]] inline Value
resolveIntVariant(OpBuilder& builder, Location loc,
                  const std::variant<int64_t, Value>& variant) {
  if (const auto* value = std::get_if<Value>(&variant)) {
    return *value;
  }
  return LLVM::ConstantOp::create(
             builder, loc, builder.getI64Type(),
             builder.getIndexAttr(std::get<int64_t>(variant)))
      .getResult();
}

} // namespace mlir::qir
