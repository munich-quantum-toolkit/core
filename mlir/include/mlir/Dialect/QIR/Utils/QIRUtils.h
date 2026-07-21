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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlir {
class OpBuilder;
class Operation;
namespace LLVM {
class AddressOfOp;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace mlir::qir {

// QIR function names

inline constexpr auto QIR_QUBIT_ARRAY_ALLOC =
    "__quantum__rt__qubit_array_allocate";
inline constexpr auto QIR_QUBIT_ARRAY_RELEASE =
    "__quantum__rt__qubit_array_release";

inline constexpr auto QIR_QUBIT_ALLOC = "__quantum__rt__qubit_allocate";
inline constexpr auto QIR_QUBIT_RELEASE = "__quantum__rt__qubit_release";

inline constexpr auto QIR_RESULT_ARRAY_ALLOC =
    "__quantum__rt__result_array_allocate";
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

#define ADD_STANDARD_GATE(NAME_BIG, NAME_SMALL)                                \
  inline constexpr auto QIR_##NAME_BIG =                                       \
      "__quantum__qis__" #NAME_SMALL "__body";                                 \
  inline constexpr auto QIR_C##NAME_BIG =                                      \
      "__quantum__qis__c" #NAME_SMALL "__body";                                \
  inline constexpr auto QIR_CC##NAME_BIG =                                     \
      "__quantum__qis__cc" #NAME_SMALL "__body";                               \
  inline constexpr auto QIR_CCC##NAME_BIG =                                    \
      "__quantum__qis__ccc" #NAME_SMALL "__body";

ADD_STANDARD_GATE(I, i)
ADD_STANDARD_GATE(X, x)
ADD_STANDARD_GATE(Y, y)
ADD_STANDARD_GATE(Z, z)
ADD_STANDARD_GATE(H, h)
ADD_STANDARD_GATE(S, s)
ADD_STANDARD_GATE(T, t)
ADD_STANDARD_GATE(SX, sx)
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
ADD_STANDARD_GATE(RCCX, rccx)

#undef ADD_STANDARD_GATE

#define ADD_ADJOINT_GATE(NAME_BIG, NAME_SMALL)                                 \
  inline constexpr auto QIR_##NAME_BIG##_ADJ =                                 \
      "__quantum__qis__" #NAME_SMALL "__adj";                                  \
  inline constexpr auto QIR_C##NAME_BIG##_ADJ =                                \
      "__quantum__qis__c" #NAME_SMALL "__adj";                                 \
  inline constexpr auto QIR_CC##NAME_BIG##_ADJ =                               \
      "__quantum__qis__cc" #NAME_SMALL "__adj";                                \
  inline constexpr auto QIR_CCC##NAME_BIG##_ADJ =                              \
      "__quantum__qis__ccc" #NAME_SMALL "__adj";

ADD_ADJOINT_GATE(S, s)
ADD_ADJOINT_GATE(T, t)
ADD_ADJOINT_GATE(SX, sx)

#undef ADD_ADJOINT_GATE

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
      llvm::reportFatalUsageError(                                             \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
  }

DEFINE_GETTER(I)
DEFINE_GETTER(X)
DEFINE_GETTER(Y)
DEFINE_GETTER(Z)
DEFINE_GETTER(H)
DEFINE_GETTER(S)
DEFINE_GETTER(T)
DEFINE_GETTER(SX)
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
DEFINE_GETTER(RCCX)

#undef DEFINE_GETTER

#define DEFINE_ADJOINT_GETTER(NAME)                                            \
  /**                                                                          \
   * @brief Gets the QIR function name for NAME                                \
   *                                                                           \
   * @param numControls Number of control qubits                               \
   * @return The QIR function name                                             \
   */                                                                          \
  inline StringRef getFnName##NAME##DG(size_t numControls) {                   \
    switch (numControls) {                                                     \
    case 0:                                                                    \
      return QIR_##NAME##_ADJ;                                                 \
    case 1:                                                                    \
      return QIR_C##NAME##_ADJ;                                                \
    case 2:                                                                    \
      return QIR_CC##NAME##_ADJ;                                               \
    case 3:                                                                    \
      return QIR_CCC##NAME##_ADJ;                                              \
    default:                                                                   \
      llvm::reportFatalUsageError(                                             \
          "Multi-controlled with more than 3 controls are currently not "      \
          "supported");                                                        \
    }                                                                          \
  }

DEFINE_ADJOINT_GETTER(S)
DEFINE_ADJOINT_GETTER(T)
DEFINE_ADJOINT_GETTER(SX)

#undef DEFINE_ADJOINT_GETTER

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

/**
 * @brief A single classical bit within a classical register.
 */
struct Bit {
  /// Label of the register containing this bit.
  StringRef registerLabel;
  /// Size of the register containing this bit.
  int64_t registerSize = 0;
  /// Index of this bit within the register.
  int64_t registerIndex = 0;
};

/**
 * @brief A classical-bit register.
 */
struct ClassicalRegister {
  /// Label of the register (e.g., "c0").
  std::string label;
  /// Number of bits in the register.
  int64_t size = 0;
  /// Result pointer for each bit.
  SmallVector<Value> results;
  /// Whether the register should be recorded in the output.
  bool record = true;

  /**
   * @brief Access a specific bit in the register.
   * @param index The index of the bit to access
   * @return A `Bit` structure representing the specified bit
   */
  Bit operator[](int64_t index) const;
};

/**
 * @brief A non-register result.
 */
struct StaticResult {
  /// The result pointer.
  Value pointer;
  /// Whether the result is recorded as an individual (`__unnamed__`) output.
  bool record = false;
};

/**
 * @brief Emit the output-recording calls.
 *
 * @details Each classical register with `record` set produces an
 * `array_record_output` marker followed by one `result_record_output` per bit;
 * registers with `record` unset are skipped. Individually recorded results
 * follow as `__unnamed__` results.
 *
 * @param builder The builder to use
 * @param anchor An operation used to locate the enclosing module
 * @param registers The classical registers to record, in order
 * @param results Result pointers indexed by result index; those with `record`
 * set are recorded individually
 */
void emitOutputRecording(OpBuilder& builder, Operation* anchor,
                         ArrayRef<ClassicalRegister> registers,
                         const DenseMap<int64_t, StaticResult>& results);

} // namespace mlir::qir
