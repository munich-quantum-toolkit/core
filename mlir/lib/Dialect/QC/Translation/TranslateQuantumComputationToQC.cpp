/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>

namespace mlir {

using namespace qc;

namespace {

/**
 * @brief Structure to store information about quantum registers
 *
 * @details
 * Maps quantum registers from the input QuantumComputation to their
 * corresponding MLIR qubit values allocated via the QCProgramBuilder.
 */
struct QregInfo {
  const ::qc::QuantumRegister* qregPtr;
  SmallVector<Value> qubits;
};

// (register ref, localIdx)
using BitMemInfo = std::pair<QCProgramBuilder::ClassicalRegister, size_t>;
using BitIndexVec = SmallVector<BitMemInfo>;

/**
 * @brief Structure to maintain state during translation
 */
struct TranslationState {
  /// Flat vector of qubit values indexed by physical qubit index
  SmallVector<Value> qubits;

  /// Mapping from global bit index to (register, local_index)
  BitIndexVec bitMap;

  /// Flat vector of measurement results
  SmallVector<Value> results;

  /// Whether the translation is currently processing an IfElseOperation
  bool inIfElse = false;

  /// Whether the translation is currently within a control modifier
  bool inCtrlOp = false;

  /// Mapping from physical qubit index to block argument
  DenseMap<size_t, Value> targetArgs;

  /// Control qubits of the current CompoundOperation
  DenseSet<::qc::Qubit> compoundControls;

  [[nodiscard]] Value getQubit(size_t index) const {
    if (inCtrlOp) {
      auto it = targetArgs.find(index);
      if (it == targetArgs.end()) {
        llvm::reportFatalInternalError("Qubit index out of bounds");
      }
      return it->second;
    }

    if (index >= qubits.size()) {
      llvm::reportFatalInternalError("Qubit index out of bounds");
    }
    return qubits[index];
  };
};

} // namespace

/**
 * @brief Allocates quantum registers using the QCProgramBuilder
 *
 * @details
 * Processes all quantum and ancilla registers from the QuantumComputation,
 * sorting them by start index, and allocates them using the builder's
 * allocQubitRegister method which generates qc.alloc operations with
 * proper register metadata.
 *
 * @param builder The QCProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @return Vector containing information about all quantum registers
 */
static SmallVector<QregInfo>
allocateQregs(QCProgramBuilder& builder,
              const ::qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  SmallVector<const ::qc::QuantumRegister*> qregPtrs;
  qregPtrs.reserve(quantumComputation.getQuantumRegisters().size() +
                   quantumComputation.getAncillaRegisters().size());
  for (const auto& qreg :
       quantumComputation.getQuantumRegisters() | std::views::values) {
    qregPtrs.emplace_back(&qreg);
  }
  for (const auto& qreg :
       quantumComputation.getAncillaRegisters() | std::views::values) {
    qregPtrs.emplace_back(&qreg);
  }

  // Sort by start index
  std::ranges::sort(qregPtrs, [](const ::qc::QuantumRegister* a,
                                 const ::qc::QuantumRegister* b) {
    return a->getStartIndex() < b->getStartIndex();
  });

  // Allocate quantum registers using the builder
  SmallVector<QregInfo> qregs;
  for (const auto* qregPtr : qregPtrs) {
    auto qubitRegister =
        builder.allocQubitRegister(static_cast<int64_t>(qregPtr->getSize()));
    qregs.emplace_back(qregPtr, std::move(qubitRegister.qubits));
  }

  return qregs;
}

/**
 * @brief Builds a flat mapping from global qubit index to qubit value
 *
 * @details
 * Creates a flat array where each index corresponds to a physical qubit
 * index, and the value is the MLIR Value representing that qubit. This
 * simplifies operation translation by providing direct qubit lookup.
 *
 * @param quantumComputation The quantum computation being translated
 * @param qregs Vector containing information about all quantum registers
 * @return Flat vector of qubit values indexed by physical qubit index
 */
static SmallVector<Value>
buildQubitMap(const ::qc::QuantumComputation& quantumComputation,
              const SmallVector<QregInfo>& qregs) {
  SmallVector<Value> flatQubits;
  const auto maxPhys = quantumComputation.getHighestPhysicalQubitIndex();
  flatQubits.resize(static_cast<size_t>(maxPhys) + 1);

  for (const auto& qreg : qregs) {
    for (size_t i = 0; i < qreg.qregPtr->getSize(); ++i) {
      const auto globalIdx = qreg.qregPtr->getStartIndex() + i;
      flatQubits[globalIdx] = qreg.qubits[i];
    }
  }

  return flatQubits;
}

/**
 * @brief Allocates classical registers using the QCProgramBuilder
 *
 * @details
 * Creates classical bit registers and builds a mapping from global classical
 * bit indices to (register, local_index) pairs. This is used for measurement
 * result storage.
 *
 * @param builder The QCProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @return Vector mapping global bit indices to register and local indices
 */
static BitIndexVec
allocateClassicalRegisters(QCProgramBuilder& builder,
                           const ::qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  SmallVector<const ::qc::ClassicalRegister*> cregPtrs;
  cregPtrs.reserve(quantumComputation.getClassicalRegisters().size());
  for (const auto& reg :
       quantumComputation.getClassicalRegisters() | std::views::values) {
    cregPtrs.emplace_back(&reg);
  }

  // Sort by start index
  std::ranges::sort(cregPtrs, [](const ::qc::ClassicalRegister* a,
                                 const ::qc::ClassicalRegister* b) {
    return a->getStartIndex() < b->getStartIndex();
  });

  // Build mapping using the builder
  BitIndexVec bitMap;
  bitMap.resize(quantumComputation.getNcbits());
  for (const auto* reg : cregPtrs) {
    const auto& mem = builder.allocClassicalBitRegister(
        static_cast<int64_t>(reg->getSize()), reg->getName());
    for (size_t i = 0; i < reg->getSize(); ++i) {
      const auto globalIdx = static_cast<size_t>(reg->getStartIndex() + i);
      bitMap[globalIdx] = {mem, i};
    }
  }

  return bitMap;
}

/**
 * @brief Adds measurement operations for a single operation
 *
 * @details
 * Translates measurement operations from the QuantumComputation to
 * qc.measure operations, storing results in classical registers.
 *
 * @param builder The QCProgramBuilder used to create operations
 * @param operation The measurement operation to translate
 * @param state The translation state
 */
static void addMeasureOp(QCProgramBuilder& builder,
                         const ::qc::Operation& operation,
                         TranslationState& state) {
  if (state.inIfElse) {
    llvm::reportFatalInternalError(
        "Measurement operations inside IfElseOperations cannot be translated "
        "to QC at the moment");
  }

  const auto& measureOp =
      dynamic_cast<const ::qc::NonUnitaryOperation&>(operation);
  const auto& targets = measureOp.getTargets();
  const auto& classics = measureOp.getClassics();

  for (size_t i = 0; i < targets.size(); ++i) {
    const auto& qubit = state.getQubit(targets[i]);
    const auto bitIdx = static_cast<size_t>(classics[i]);
    const auto& [mem, localIdx] = state.bitMap[bitIdx];
    const auto& bit = mem[static_cast<int64_t>(localIdx)];
    state.results[bitIdx] = builder.measure(qubit, bit);
  }
}

/**
 * @brief Adds reset operations for a single operation
 *
 * @details
 * Translates reset operations from the QuantumComputation to
 * qc.reset operations.
 *
 * @param builder The QCProgramBuilder used to create operations
 * @param operation The reset operation to translate
 * @param state The translation state
 */
static void addResetOp(QCProgramBuilder& builder,
                       const ::qc::Operation& operation,
                       TranslationState& state) {
  for (const auto& target : operation.getTargets()) {
    auto qubit = state.getQubit(target);
    builder.reset(qubit);
  }
}

/**
 * @brief Extracts positive control qubits from an operation
 *
 * @details
 * Iterates through the controls of the given operation and collects
 * the qubit values corresponding to positive controls.
 *
 * @param operation The operation containing controls
 * @param state The translation state
 * @return Vector of qubit values corresponding to positive controls
 */
static SmallVector<Value> getControls(const ::qc::Operation& operation,
                                      TranslationState& state) {
  SmallVector<Value> controls;
  for (const auto& [control, type] : operation.getControls()) {
    if (state.compoundControls.contains(control)) {
      continue;
    }
    if (type == ::qc::Control::Type::Neg) {
      llvm::reportFatalInternalError(
          "Negative controls cannot be translated to QC at the moment");
    }
    controls.push_back(state.getQubit(control));
  }
  return controls;
}

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CORE, OP_QC)                       \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& target = state.getQubit(operation.getTargets()[0]);            \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(target);                                                   \
    } else {                                                                   \
      builder.mc##OP_QC(controls, target);                                     \
    }                                                                          \
  }

DEFINE_ONE_TARGET_ZERO_PARAMETER(I, id)
DEFINE_ONE_TARGET_ZERO_PARAMETER(X, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(Y, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(Z, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(H, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(S, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(Sdg, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(T, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(Tdg, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SX, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdg, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CORE, OP_QC)                        \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& param = operation.getParameter()[0];                           \
    const auto& target = state.getQubit(operation.getTargets()[0]);            \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(param, target);                                            \
    } else {                                                                   \
      builder.mc##OP_QC(param, controls, target);                              \
    }                                                                          \
  }

DEFINE_ONE_TARGET_ONE_PARAMETER(RX, rx)
DEFINE_ONE_TARGET_ONE_PARAMETER(RY, ry)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZ, rz)
DEFINE_ONE_TARGET_ONE_PARAMETER(P, p)

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CORE, OP_QC)                        \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& param1 = operation.getParameter()[0];                          \
    const auto& param2 = operation.getParameter()[1];                          \
    const auto& target = state.getQubit(operation.getTargets()[0]);            \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(param1, param2, target);                                   \
    } else {                                                                   \
      builder.mc##OP_QC(param1, param2, controls, target);                     \
    }                                                                          \
  }

DEFINE_ONE_TARGET_TWO_PARAMETER(R, r)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2, u2)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CORE, OP_QC)                      \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& param1 = operation.getParameter()[0];                          \
    const auto& param2 = operation.getParameter()[1];                          \
    const auto& param3 = operation.getParameter()[2];                          \
    const auto& target = state.getQubit(operation.getTargets()[0]);            \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(param1, param2, param3, target);                           \
    } else {                                                                   \
      builder.mc##OP_QC(param1, param2, param3, controls, target);             \
    }                                                                          \
  }

DEFINE_ONE_TARGET_THREE_PARAMETER(U, u)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CORE, OP_QC)                       \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& target0 = state.getQubit(operation.getTargets()[0]);           \
    const auto& target1 = state.getQubit(operation.getTargets()[1]);           \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(target0, target1);                                         \
    } else {                                                                   \
      builder.mc##OP_QC(controls, target0, target1);                           \
    }                                                                          \
  }

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAP, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAP, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCX, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECR, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

static void addISWAPdgOp(QCProgramBuilder& builder,
                         const ::qc::Operation& operation,
                         TranslationState& state) {
  auto target0 = state.getQubit(operation.getTargets()[0]);
  auto target1 = state.getQubit(operation.getTargets()[1]);
  if (const auto& controls = getControls(operation, state); controls.empty()) {
    builder.inv({target0, target1}, [&](ValueRange qubits) {
      builder.iswap(qubits[0], qubits[1]);
    });
  } else {
    builder.ctrl(controls, {target0, target1}, [&](ValueRange targets) {
      builder.inv(targets, [&](ValueRange qubits) {
        builder.iswap(qubits[0], qubits[1]);
      });
    });
  }
}

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CORE, OP_QC)                        \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& param = operation.getParameter()[0];                           \
    const auto& target0 = state.getQubit(operation.getTargets()[0]);           \
    const auto& target1 = state.getQubit(operation.getTargets()[1]);           \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(param, target0, target1);                                  \
    } else {                                                                   \
      builder.mc##OP_QC(param, controls, target0, target1);                    \
    }                                                                          \
  }

DEFINE_TWO_TARGET_ONE_PARAMETER(RXX, rxx)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYY, ryy)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZX, rzx)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZ, rzz)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CORE, OP_QC)                        \
  /**                                                                          \
   * @brief Adds a qc.OP_QC operation                                          \
   *                                                                           \
   * @details                                                                  \
   * Translates an OP_CORE operation from the QuantumComputation to            \
   * qc.OP_QC.                                                                 \
   *                                                                           \
   * @param builder The QCProgramBuilder used to create operations             \
   * @param operation The OP_CORE operation to translate                       \
   * @param state The translation state                                        \
   */                                                                          \
  static void add##OP_CORE##Op(QCProgramBuilder& builder,                      \
                               const ::qc::Operation& operation,               \
                               TranslationState& state) {                      \
    const auto& param1 = operation.getParameter()[0];                          \
    const auto& param2 = operation.getParameter()[1];                          \
    const auto& target0 = state.getQubit(operation.getTargets()[0]);           \
    const auto& target1 = state.getQubit(operation.getTargets()[1]);           \
    if (const auto& controls = getControls(operation, state);                  \
        controls.empty()) {                                                    \
      builder.OP_QC(param1, param2, target0, target1);                         \
    } else {                                                                   \
      builder.mc##OP_QC(param1, param2, controls, target0, target1);           \
    }                                                                          \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER(XXplusYY, xx_plus_yy)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXminusYY, xx_minus_yy)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

static void addBarrierOp(QCProgramBuilder& builder,
                         const ::qc::Operation& operation,
                         TranslationState& state) {
  SmallVector<Value> targets;
  for (const auto& targetIdx : operation.getTargets()) {
    targets.push_back(state.getQubit(targetIdx));
  }
  builder.barrier(targets);
}

// Forward declaration
static LogicalResult translateOperation(QCProgramBuilder& builder,
                                        const ::qc::Operation& operation,
                                        TranslationState& state);

// CompoundOp

static LogicalResult addCompoundOp(QCProgramBuilder& builder,
                                   const ::qc::Operation& operation,
                                   TranslationState& state) {
  const auto& compoundOp =
      dynamic_cast<const ::qc::CompoundOperation&>(operation);
  if (const auto& controls = getControls(operation, state); controls.empty()) {
    for (const auto& op : compoundOp) {
      if (failed(translateOperation(builder, *op, state))) {
        return failure();
      }
    }
  } else {
    // Collect targets
    DenseMap<uint32_t, Value> targetMap;
    for (const auto& op : compoundOp) {
      if (dynamic_cast<const ::qc::CompoundOperation*>(op.get()) != nullptr) {
        llvm::reportFatalInternalError("Nested CompoundOperations cannot be "
                                       "translated to QC at the moment");
      }
      for (const auto& target : op->getTargets()) {
        if (!targetMap.contains(target)) {
          targetMap[target] = state.getQubit(target);
        }
      }
      for (const auto& control : op->getControls()) {
        if (compoundOp.getControls().contains(control)) {
          continue;
        }
        const auto& qubit = control.qubit;
        if (!targetMap.contains(qubit)) {
          targetMap[qubit] = state.getQubit(qubit);
        }
      }
    }
    for (const auto& [control, _] : compoundOp.getControls()) {
      state.compoundControls.insert(control);
    }
    SmallVector<std::pair<uint32_t, Value>> sortedPairs(targetMap.begin(),
                                                        targetMap.end());
    llvm::sort(sortedPairs.begin(), sortedPairs.end(),
               [](const auto& a, const auto& b) { return a.first < b.first; });
    SmallVector<Value> targets;
    for (const auto& pair : sortedPairs) {
      targets.push_back(pair.second);
    }
    // Build control modifier
    builder.ctrl(controls, targets, [&](ValueRange targetArgs) {
      state.inCtrlOp = true;
      for (size_t i = 0; i < sortedPairs.size(); ++i) {
        state.targetArgs[sortedPairs[i].first] = targetArgs[i];
      }
      for (const auto& op : compoundOp) {
        if (failed(translateOperation(builder, *op, state))) {
          llvm::reportFatalInternalError("Failed to translate operation inside "
                                         "controlled CompoundOperation");
        }
      }
      state.targetArgs.clear();
      state.inCtrlOp = false;
    });
  }
  return success();
}

// IfElseOp

static LogicalResult addIfElseOp(QCProgramBuilder& builder,
                                 const ::qc::Operation& operation,
                                 TranslationState& state) {
  const auto& ifElse = dynamic_cast<const ::qc::IfElseOperation&>(operation);

  if (ifElse.getControlRegister().has_value()) {
    llvm::errs() << "IfElseOperations controlled by registers cannot be "
                    "translated to QC at the moment\n";
    return failure();
  }

  assert(ifElse.getControlBit().has_value());
  const auto bitIdx = static_cast<size_t>(*ifElse.getControlBit());
  auto controlValue = state.results[bitIdx];
  if (controlValue == nullptr) {
    llvm::errs() << "Control bit does not contain a measurement result\n";
    return failure();
  }
  auto expectedValue = builder.boolConstant(ifElse.getExpectedValueBit());

  // Define comparison predicate
  const auto comparisonKind = ifElse.getComparisonKind();
  auto predicate = arith::CmpIPredicate::eq;
  switch (comparisonKind) {
  case ::qc::ComparisonKind::Eq:
    predicate = arith::CmpIPredicate::eq;
    break;
  case ::qc::ComparisonKind::Neq:
    predicate = arith::CmpIPredicate::ne;
    break;
  default:
    llvm::errs() << "Unsupported comparison kind in IfElseOperation\n";
    return failure();
  }

  // Define condition
  auto condition =
      arith::CmpIOp::create(builder, predicate, controlValue, expectedValue);

  // Define if-else operation
  auto thenResult = success();
  auto thenBuilder = [&] {
    state.inIfElse = true;
    thenResult = translateOperation(builder, *ifElse.getThenOp(), state);
    state.inIfElse = false;
  };

  auto elseResult = success();
  auto elseBuilder = [&] {
    state.inIfElse = true;
    elseResult = translateOperation(builder, *ifElse.getElseOp(), state);
    state.inIfElse = false;
  };

  if (ifElse.getElseOp() != nullptr) {
    builder.scfIf(condition.getResult(), thenBuilder, elseBuilder);
  } else {
    builder.scfIf(condition.getResult(), thenBuilder);
  }

  if (failed(thenResult)) {
    llvm::errs() << "Failed to translate then branch of IfElseOperation\n";
    return failure();
  }
  if (failed(elseResult)) {
    llvm::errs() << "Failed to translate else branch of IfElseOperation\n";
    return failure();
  }

  return success();
}

#define ADD_OP_CASE(OP_CORE)                                                   \
  case ::qc::OpType::OP_CORE:                                                  \
    add##OP_CORE##Op(builder, operation, state);                               \
    return success();

/**
 * @brief Translates an operation from QuantumComputation to QC dialect
 *
 * @param builder The QCProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @param state The translation state
 * @return Success if all supported operations were translated
 */
static LogicalResult translateOperation(QCProgramBuilder& builder,
                                        const ::qc::Operation& operation,
                                        TranslationState& state) {
  switch (operation.getType()) {
  case ::qc::OpType::Measure:
    addMeasureOp(builder, operation, state);
    return success();
    ADD_OP_CASE(Reset)
    ADD_OP_CASE(I)
    ADD_OP_CASE(X)
    ADD_OP_CASE(Y)
    ADD_OP_CASE(Z)
    ADD_OP_CASE(H)
    ADD_OP_CASE(S)
    ADD_OP_CASE(Sdg)
    ADD_OP_CASE(T)
    ADD_OP_CASE(Tdg)
    ADD_OP_CASE(SX)
    ADD_OP_CASE(SXdg)
    ADD_OP_CASE(RX)
    ADD_OP_CASE(RY)
    ADD_OP_CASE(RZ)
    ADD_OP_CASE(P)
    ADD_OP_CASE(R)
    ADD_OP_CASE(U2)
    ADD_OP_CASE(U)
    ADD_OP_CASE(SWAP)
    ADD_OP_CASE(iSWAP)
    ADD_OP_CASE(DCX)
    ADD_OP_CASE(ECR)
    ADD_OP_CASE(RXX)
    ADD_OP_CASE(RYY)
    ADD_OP_CASE(RZX)
    ADD_OP_CASE(RZZ)
    ADD_OP_CASE(XXplusYY)
    ADD_OP_CASE(XXminusYY)
    ADD_OP_CASE(Barrier)
  case ::qc::OpType::iSWAPdg:
    addISWAPdgOp(builder, operation, state);
    return success();
  case ::qc::OpType::Compound:
    if (failed(addCompoundOp(builder, operation, state))) {
      return failure();
    }
    return success();
  case ::qc::OpType::IfElse:
    if (failed(addIfElseOp(builder, operation, state))) {
      return failure();
    }
    return success();
  default:
    llvm::errs() << operation.getName() << " cannot be translated to QC\n";
    return failure();
  }
}

#undef ADD_OP_CASE

/**
 * @brief Translates operations from QuantumComputation to QC dialect
 *
 * @details
 * Iterates through all operations in the QuantumComputation and translates
 * them to QC operations.
 *
 * @param builder The QCProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @param state The translation state
 * @return Success if all supported operations were translated
 */
static LogicalResult
translateOperations(QCProgramBuilder& builder,
                    const ::qc::QuantumComputation& quantumComputation,
                    TranslationState& state) {
  if (quantumComputation.hasGlobalPhase()) {
    builder.gphase(quantumComputation.getGlobalPhase());
  }
  for (const auto& operation : quantumComputation) {
    if (translateOperation(builder, *operation, state).failed()) {
      llvm::errs() << "Failed to translate operation: " << operation->getName()
                   << "\n";
      return failure();
    }
  }
  return success();
}

/**
 * @brief Translates a QuantumComputation to an MLIR module with QC
 * operations
 *
 * @details
 * This function takes a quantum computation and translates it into an MLIR
 * module containing QC operations. It uses the QCProgramBuilder to
 * handle module and function creation, resource allocation, and operation
 * translation.
 *
 * The translation process:
 * 1. Creates a QCProgramBuilder and initializes it (creates the main
 * function)
 * 2. Allocates quantum registers using qc.alloc
 * 3. Tracks classical registers for measurement results
 * 4. Translates operations
 * 5. Finalizes the module (adds return statement with exit code 0)
 *
 * If the translation fails due to an unsupported operation, a fatal error is
 * reported.
 *
 * @param context The MLIR context in which the module will be created
 * @param quantumComputation The quantum computation to translate
 * @return OwningOpRef containing the translated MLIR module
 */
OwningOpRef<ModuleOp> translateQuantumComputationToQC(
    MLIRContext* context, const ::qc::QuantumComputation& quantumComputation) {
  // Create and initialize the builder (creates module and main function)
  QCProgramBuilder builder(context);
  SmallVector<Type> resultTypes(quantumComputation.getNcbits());
  for (auto i = 0; i < quantumComputation.getNcbits(); ++i) {
    resultTypes[i] = builder.getI1Type();
  }
  if (quantumComputation.getNcbits() == 0) {
    // Without classical bits, we instead return an exit code 0.
    resultTypes.push_back(builder.getI64Type());
  }
  builder.initialize(resultTypes);

  // Allocate quantum registers using the builder
  const auto qregs = allocateQregs(builder, quantumComputation);

  // Build flat qubit mapping for easy lookup
  const auto qubits = buildQubitMap(quantumComputation, qregs);

  // Allocate classical registers using the builder
  const auto bitMap = allocateClassicalRegisters(builder, quantumComputation);

  // Allocate result map
  SmallVector<Value> results(quantumComputation.getNcbits(), nullptr);

  TranslationState state{.qubits = qubits,
                         .bitMap = bitMap,
                         .results = std::move(results),
                         .targetArgs = DenseMap<size_t, Value>{}};

  // Translate operations
  if (translateOperations(builder, quantumComputation, state).failed()) {
    llvm::reportFatalInternalError(
        "Failed to translate QuantumComputation to QC");
  }

  // Finalize and return the module (adds return statement and transfers
  // ownership)
  return quantumComputation.getNcbits() == 0
             ? builder.finalize({builder.intConstant(0)})
             : builder.finalize(state.results);
}

} // namespace mlir
