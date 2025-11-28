/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/Translation/TranslateQuantumComputationToQuartz.h"

#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <ranges>
#include <utility>

namespace mlir {

using namespace quartz;

namespace {

/**
 * @brief Structure to store information about quantum registers
 *
 * @details
 * Maps quantum registers from the input QuantumComputation to their
 * corresponding MLIR qubit values allocated via the QuartzProgramBuilder.
 */
struct QregInfo {
  const qc::QuantumRegister* qregPtr;
  llvm::SmallVector<Value> qubits;
};

using BitMemInfo = std::pair<QuartzProgramBuilder::ClassicalRegister*,
                             size_t>; // (register ref, localIdx)
using BitIndexVec = llvm::SmallVector<BitMemInfo>;

/**
 * @brief Allocates quantum registers using the QuartzProgramBuilder
 *
 * @details
 * Processes all quantum and ancilla registers from the QuantumComputation,
 * sorting them by start index, and allocates them using the builder's
 * allocQubitRegister method which generates quartz.alloc operations with
 * proper register metadata.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @return Vector containing information about all quantum registers
 */
llvm::SmallVector<QregInfo>
allocateQregs(QuartzProgramBuilder& builder,
              const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::QuantumRegister*> qregPtrs;
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
  std::ranges::sort(
      qregPtrs, [](const qc::QuantumRegister* a, const qc::QuantumRegister* b) {
        return a->getStartIndex() < b->getStartIndex();
      });

  // Allocate quantum registers using the builder
  llvm::SmallVector<QregInfo> qregs;
  for (const auto* qregPtr : qregPtrs) {
    auto qubits = builder.allocQubitRegister(
        static_cast<int64_t>(qregPtr->getSize()), qregPtr->getName());
    qregs.emplace_back(qregPtr, std::move(qubits));
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
llvm::SmallVector<Value>
buildQubitMap(const qc::QuantumComputation& quantumComputation,
              const llvm::SmallVector<QregInfo>& qregs) {
  llvm::SmallVector<Value> flatQubits;
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
 * @brief Allocates classical registers using the QuartzProgramBuilder
 *
 * @details
 * Creates classical bit registers and builds a mapping from global classical
 * bit indices to (register, local_index) pairs. This is used for measurement
 * result storage.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @return Vector mapping global bit indices to register and local indices
 */
BitIndexVec
allocateClassicalRegisters(QuartzProgramBuilder& builder,
                           const qc::QuantumComputation& quantumComputation) {
  // Build list of pointers for sorting
  llvm::SmallVector<const qc::ClassicalRegister*> cregPtrs;
  cregPtrs.reserve(quantumComputation.getClassicalRegisters().size());
  for (const auto& reg :
       quantumComputation.getClassicalRegisters() | std::views::values) {
    cregPtrs.emplace_back(&reg);
  }

  // Sort by start index
  std::ranges::sort(cregPtrs, [](const qc::ClassicalRegister* a,
                                 const qc::ClassicalRegister* b) {
    return a->getStartIndex() < b->getStartIndex();
  });

  // Build mapping using the builder
  BitIndexVec bitMap;
  bitMap.resize(quantumComputation.getNcbits());
  for (const auto* reg : cregPtrs) {
    auto& mem = builder.allocClassicalBitRegister(
        static_cast<int64_t>(reg->getSize()), reg->getName());
    for (size_t i = 0; i < reg->getSize(); ++i) {
      const auto globalIdx = static_cast<size_t>(reg->getStartIndex() + i);
      bitMap[globalIdx] = {&mem, i};
    }
  }

  return bitMap;
}

/**
 * @brief Adds measurement operations for a single operation
 *
 * @details
 * Translates measurement operations from the QuantumComputation to
 * quartz.measure operations, storing results in classical registers.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The measurement operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 * @param bitMap Mapping from global bit index to (register, local_index)
 */
void addMeasureOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
                  const llvm::SmallVector<Value>& qubits,
                  const BitIndexVec& bitMap) {
  const auto& measureOp =
      dynamic_cast<const qc::NonUnitaryOperation&>(operation);
  const auto& targets = measureOp.getTargets();
  const auto& classics = measureOp.getClassics();

  for (size_t i = 0; i < targets.size(); ++i) {
    const auto& qubit = qubits[targets[i]];
    const auto bitIdx = static_cast<size_t>(classics[i]);
    const auto& [mem, localIdx] = bitMap[bitIdx];

    // Use builder's measure method which keeps output record
    builder.measure(qubit, (*mem)[static_cast<int64_t>(localIdx)]);
  }
}

/**
 * @brief Adds reset operations for a single operation
 *
 * @details
 * Translates reset operations from the QuantumComputation to
 * quartz.reset operations.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The reset operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addResetOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
                const llvm::SmallVector<Value>& qubits) {
  for (const auto& target : operation.getTargets()) {
    const Value qubit = qubits[target];
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
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 * @return Vector of qubit values corresponding to positive controls
 */
llvm::SmallVector<Value>
getPosControls(const qc::Operation& operation,
               const llvm::SmallVector<Value>& qubits) {
  llvm::SmallVector<Value> controls;
  for (const auto& [control, type] : operation.getControls()) {
    if (type == qc::Control::Type::Neg) {
      continue;
    }
    controls.push_back(qubits[control]);
  }
  return controls;
}

/**
 * @brief Adds an Id operation
 *
 * @details
 * Translates an Id operation from the QuantumComputation to quartz.id.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The Id operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addIdOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.id(target);
  } else {
    builder.mcid(posControls, target);
  }
}

/**
 * @brief Adds an X operation
 *
 * @details
 * Translates an X operation from the QuantumComputation to quartz.x.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The X operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.x(target);
  } else {
    builder.mcx(posControls, target);
  }
}

/**
 * @brief Adds a Y operation
 *
 * @details
 * Translates a Y operation from the QuantumComputation to quartz.y.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The Y operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addYOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.y(target);
  } else {
    builder.mcy(posControls, target);
  }
}

/**
 * @brief Adds a Z operation
 *
 * @details
 * Translates a Z operation from the QuantumComputation to quartz.z.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The Z operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addZOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.z(target);
  } else {
    builder.mcz(posControls, target);
  }
}

/**
 * @brief Adds an H operation
 *
 * @details
 * Translates an H operation from the QuantumComputation to quartz.h.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The H operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addHOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.h(target);
  } else {
    builder.mch(posControls, target);
  }
}

/**
 * @brief Adds an S operation
 *
 * @details
 * Translates an S operation from the QuantumComputation to quartz.s.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The S operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addSOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.s(target);
  } else {
    builder.mcs(posControls, target);
  }
}

/**
 * @brief Adds an Sdg operation
 *
 * @details
 * Translates an Sdg operation from the QuantumComputation to quartz.sdg.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The Sdg operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addSdgOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.sdg(target);
  } else {
    builder.mcsdg(posControls, target);
  }
}

/**
 * @brief Adds a T operation
 *
 * @details
 * Translates a T operation from the QuantumComputation to quartz.t.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The T operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addTOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.t(target);
  } else {
    builder.mct(posControls, target);
  }
}

/**
 * @brief Adds a Tdg operation
 *
 * @details
 * Translates a Tdg operation from the QuantumComputation to quartz.tdg.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The Tdg operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addTdgOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.tdg(target);
  } else {
    builder.mctdg(posControls, target);
  }
}

/**
 * @brief Adds an SX operation
 *
 * @details
 * Translates an SX operation from the QuantumComputation to quartz.sx.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The SX operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addSXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.sx(target);
  } else {
    builder.mcsx(posControls, target);
  }
}

/**
 * @brief Adds an SXdg operation
 *
 * @details
 * Translates an SXdg operation from the QuantumComputation to quartz.sxdg.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The SXdg operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addSXdgOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
               const llvm::SmallVector<Value>& qubits) {
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.sxdg(target);
  } else {
    builder.mcsxdg(posControls, target);
  }
}

/**
 * @brief Adds an RX operation
 *
 * @details
 * Translates an RX operation from the QuantumComputation to quartz.rx.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RX operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.rx(theta, target);
  } else {
    builder.mcrx(theta, posControls, target);
  }
}

/**
 * @brief Adds an RY operation
 *
 * @details
 * Translates an RY operation from the QuantumComputation to quartz.ry.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RY operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRYOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.ry(theta, target);
  } else {
    builder.mcry(theta, posControls, target);
  }
}

/**
 * @brief Adds an RZ operation
 *
 * @details
 * Translates an RZ operation from the QuantumComputation to quartz.rz.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RZ operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRZOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.rz(theta, target);
  } else {
    builder.mcrz(theta, posControls, target);
  }
}

/**
 * @brief Adds a P operation
 *
 * @details
 * Translates a P operation from the QuantumComputation to quartz.p.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The P operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addPOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.p(theta, target);
  } else {
    builder.mcp(theta, posControls, target);
  }
}

/**
 * @brief Adds an R operation
 *
 * @details
 * Translates an R operation from the QuantumComputation to quartz.r.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The R operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addROp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& phi = operation.getParameter()[1];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.r(theta, phi, target);
  } else {
    builder.mcr(theta, phi, posControls, target);
  }
}

/**
 * @brief Adds a U2 operation
 *
 * @details
 * Translate a U2 operation from the QuantumComputation to quartz.u2.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The U2 operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addU2Op(QuartzProgramBuilder& builder, const qc::Operation& operation,
             const llvm::SmallVector<Value>& qubits) {
  const auto& phi = operation.getParameter()[0];
  const auto& lambda = operation.getParameter()[1];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.u2(phi, lambda, target);
  } else {
    builder.mcu2(phi, lambda, posControls, target);
  }
}

/**
 * @brief Adds a U operation
 *
 * @details
 * Translate a U operation from the QuantumComputation to quartz.u.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The U operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addUOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
            const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& phi = operation.getParameter()[1];
  const auto& lambda = operation.getParameter()[2];
  const auto& target = qubits[operation.getTargets()[0]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.u(theta, phi, lambda, target);
  } else {
    builder.mcu(theta, phi, lambda, posControls, target);
  }
}

/**
 * @brief Adds a SWAP operation
 *
 * @details
 * Translate a SWAP operation from the QuantumComputation to quartz.swap.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The SWAP operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addSWAPOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
               const llvm::SmallVector<Value>& qubits) {
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.swap(target0, target1);
  } else {
    builder.mcswap(posControls, target0, target1);
  }
}

/**
 * @brief Adds an iSWAP operation
 *
 * @details
 * Translate an iSWAP operation from the QuantumComputation to quartz.iswap.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The iSWAP operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addiSWAPOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
                const llvm::SmallVector<Value>& qubits) {
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.iswap(target0, target1);
  } else {
    builder.mciswap(posControls, target0, target1);
  }
}

/**
 * @brief Adds a DCX operation
 *
 * @details
 * Translate a DCX operation from the QuantumComputation to quartz.dcx.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The DCX operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addDCXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.dcx(target0, target1);
  } else {
    builder.mcdcx(posControls, target0, target1);
  }
}

/**
 * @brief Adds an ECR operation
 *
 * @details
 * Translate an ECR operation from the QuantumComputation to quartz.ecr.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The ECR operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addECROp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.ecr(target0, target1);
  } else {
    builder.mcecr(posControls, target0, target1);
  }
}

/**
 * @brief Adds an RXX operation
 *
 * @details
 * Translate an RXX operation from the QuantumComputation to quartz.rxx.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RXX operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRXXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.rxx(theta, target0, target1);
  } else {
    builder.mcrxx(theta, posControls, target0, target1);
  }
}

/**
 * @brief Adds an RYY operation
 *
 * @details
 * Translate an RYY operation from the QuantumComputation to quartz.ryy.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RYY operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRYYOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.ryy(theta, target0, target1);
  } else {
    builder.mcryy(theta, posControls, target0, target1);
  }
}

/**
 * @brief Adds an RZX operation
 *
 * @details
 * Translate an RZX operation from the QuantumComputation to quartz.rzx.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RZX operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRZXOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.rzx(theta, target0, target1);
  } else {
    builder.mcrzx(theta, posControls, target0, target1);
  }
}

/**
 * @brief Adds an RZZ operation
 *
 * @details
 * Translate an RZZ operation from the QuantumComputation to quartz.rzz.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The RZZ operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addRZZOp(QuartzProgramBuilder& builder, const qc::Operation& operation,
              const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.rzz(theta, target0, target1);
  } else {
    builder.mcrzz(theta, posControls, target0, target1);
  }
}

/**
 * @brief Adds an XX+YY operation
 *
 * @details
 * Translate an XX+YY operation from the QuantumComputation to
 * quartz.xx_plus_yy.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The XX+YY operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addXXPlusYYOp(QuartzProgramBuilder& builder,
                   const qc::Operation& operation,
                   const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& beta = operation.getParameter()[1];
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.xx_plus_yy(theta, beta, target0, target1);
  } else {
    builder.mcxx_plus_yy(theta, beta, posControls, target0, target1);
  }
}

/**
 * @brief Adds an XX-YY operation
 *
 * @details
 * Translate an XX-YY operation from the QuantumComputation to
 * quartz.xx_minus_yy.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param operation The XX-YY operation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 */
void addXXMinusYYOp(QuartzProgramBuilder& builder,
                    const qc::Operation& operation,
                    const llvm::SmallVector<Value>& qubits) {
  const auto& theta = operation.getParameter()[0];
  const auto& beta = operation.getParameter()[1];
  const auto& target0 = qubits[operation.getTargets()[0]];
  const auto& target1 = qubits[operation.getTargets()[1]];
  if (const auto& posControls = getPosControls(operation, qubits);
      posControls.empty()) {
    builder.xx_minus_yy(theta, beta, target0, target1);
  } else {
    builder.mcxx_minus_yy(theta, beta, posControls, target0, target1);
  }
}

/**
 * @brief Translates operations from QuantumComputation to Quartz dialect
 *
 * @details
 * Iterates through all operations in the QuantumComputation and translates
 * them to Quartz dialect operations. Currently supports:
 * - Measurement operations
 * - Reset operations
 *
 * Unary gates and other operations will be added as the Quartz dialect
 * is expanded.
 *
 * @param builder The QuartzProgramBuilder used to create operations
 * @param quantumComputation The quantum computation to translate
 * @param qubits Flat vector of qubit values indexed by physical qubit index
 * @param bitMap Mapping from global bit index to (register, local_index)
 * @return Success if all supported operations were translated
 */
LogicalResult
translateOperations(QuartzProgramBuilder& builder,
                    const qc::QuantumComputation& quantumComputation,
                    const llvm::SmallVector<Value>& qubits,
                    const BitIndexVec& bitMap) {
  for (const auto& operation : quantumComputation) {
    switch (operation->getType()) {
    case qc::OpType::Measure:
      addMeasureOp(builder, *operation, qubits, bitMap);
      break;
    case qc::OpType::Reset:
      addResetOp(builder, *operation, qubits);
      break;
    case qc::OpType::I:
      addIdOp(builder, *operation, qubits);
      break;
    case qc::OpType::X:
      addXOp(builder, *operation, qubits);
      break;
    case qc::OpType::Y:
      addYOp(builder, *operation, qubits);
      break;
    case qc::OpType::Z:
      addZOp(builder, *operation, qubits);
      break;
    case qc::OpType::H:
      addHOp(builder, *operation, qubits);
      break;
    case qc::OpType::S:
      addSOp(builder, *operation, qubits);
      break;
    case qc::OpType::Sdg:
      addSdgOp(builder, *operation, qubits);
      break;
    case qc::OpType::T:
      addTOp(builder, *operation, qubits);
      break;
    case qc::OpType::Tdg:
      addTdgOp(builder, *operation, qubits);
      break;
    case qc::OpType::SX:
      addSXOp(builder, *operation, qubits);
      break;
    case qc::OpType::SXdg:
      addSXdgOp(builder, *operation, qubits);
      break;
    case qc::OpType::RX:
      addRXOp(builder, *operation, qubits);
      break;
    case qc::OpType::RY:
      addRYOp(builder, *operation, qubits);
      break;
    case qc::OpType::RZ:
      addRZOp(builder, *operation, qubits);
      break;
    case qc::OpType::P:
      addPOp(builder, *operation, qubits);
      break;
    case qc::OpType::R:
      addROp(builder, *operation, qubits);
      break;
    case qc::OpType::U2:
      addU2Op(builder, *operation, qubits);
      break;
    case qc::OpType::U:
      addUOp(builder, *operation, qubits);
      break;
    case qc::OpType::SWAP:
      addSWAPOp(builder, *operation, qubits);
      break;
    case qc::OpType::iSWAP:
      addiSWAPOp(builder, *operation, qubits);
      break;
    case qc::OpType::DCX:
      addDCXOp(builder, *operation, qubits);
      break;
    case qc::OpType::ECR:
      addECROp(builder, *operation, qubits);
      break;
    case qc::OpType::RXX:
      addRXXOp(builder, *operation, qubits);
      break;
    case qc::OpType::RYY:
      addRYYOp(builder, *operation, qubits);
      break;
    case qc::OpType::RZX:
      addRZXOp(builder, *operation, qubits);
      break;
    case qc::OpType::RZZ:
      addRZZOp(builder, *operation, qubits);
      break;
    case qc::OpType::XXplusYY:
      addXXPlusYYOp(builder, *operation, qubits);
      break;
    case qc::OpType::XXminusYY:
      addXXMinusYYOp(builder, *operation, qubits);
      break;
    default:
      // Unsupported operation - skip for now
      // As the Quartz dialect is expanded, more operations will be supported
      continue;
    }
  }

  return success();
}

} // namespace

/**
 * @brief Translates a QuantumComputation to an MLIR module with Quartz
 * operations
 *
 * @details
 * This function takes a quantum computation and translates it into an MLIR
 * module containing Quartz dialect operations. It uses the QuartzProgramBuilder
 * to handle module and function creation, resource allocation, and operation
 * translation.
 *
 * The translation process:
 * 1. Creates a QuartzProgramBuilder and initializes it (creates main function
 *    with signature () -> i64)
 * 2. Allocates quantum registers using quartz.alloc with register metadata
 * 3. Tracks classical registers for measurement results
 * 4. Translates operations (currently: measure, reset)
 * 5. Finalizes the module (adds return statement with exit code 0)
 *
 * The generated main function returns exit code 0 to indicate successful
 * execution of the quantum program.
 *
 * Currently supported operations:
 * - Measurement (quartz.measure)
 * - Reset (quartz.reset)
 *
 * Operations not yet supported are silently skipped. As the Quartz dialect
 * is expanded with gate operations, this translation will be enhanced.
 *
 * @param context The MLIR context in which the module will be created
 * @param quantumComputation The quantum computation to translate
 * @return OwningOpRef containing the translated MLIR module
 */
OwningOpRef<ModuleOp> translateQuantumComputationToQuartz(
    MLIRContext* context, const qc::QuantumComputation& quantumComputation) {
  // Create and initialize the builder (creates module and main function)
  QuartzProgramBuilder builder(context);
  builder.initialize();

  // Allocate quantum registers using the builder
  const auto qregs = allocateQregs(builder, quantumComputation);

  // Build flat qubit mapping for easy lookup
  const auto qubits = buildQubitMap(quantumComputation, qregs);

  // Allocate classical registers using the builder
  const auto bitMap = allocateClassicalRegisters(builder, quantumComputation);

  // Translate operations
  if (translateOperations(builder, quantumComputation, qubits, bitMap)
          .failed()) {
    // Note: Currently all operations succeed or are skipped
    // This check is here for future error handling
  }

  // Finalize and return the module (adds return statement and transfers
  // ownership)
  return builder.finalize();
}

} // namespace mlir
