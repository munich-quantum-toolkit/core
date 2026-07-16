/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"

#include "mlir/Dialect/OQ3/IR/GateCatalog.h"
#include "mlir/Dialect/OQ3/IR/OQ3Dialect.h" // IWYU pragma: associated
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;
using namespace mlir::oq3;

#include "mlir/Dialect/OQ3/IR/OQ3OpsDialect.cpp.inc"
#include "mlir/Dialect/OQ3/IR/OQ3OpsEnums.cpp.inc"

void OQ3Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OQ3/IR/OQ3Ops.cpp.inc"
      >();
}

static LogicalResult verifyGateSignature(Operation* operation,
                                         FunctionType type) {
  bool sawQubit = false;
  for (auto input : type.getInputs()) {
    if (isa<qc::QubitType>(input)) {
      sawQubit = true;
      continue;
    }
    if (!isa<IntegerType, FloatType, ComplexType>(input)) {
      return operation->emitOpError(
          "requires every input to be an OpenQASM scalar or qubit type");
    }
    if (sawQubit) {
      return operation->emitOpError(
          "requires scalar parameters to precede all qubit inputs");
    }
  }
  return success();
}

LogicalResult GateOp::verify() {
  auto type = getFunctionType();
  if (!isa<FunctionType>(type)) {
    return emitOpError("requires a function type");
  }
  auto functionType = cast<FunctionType>(type);
  if (getBody().empty()) {
    return emitOpError("requires a body");
  }
  auto& entry = getBody().front();
  if (entry.getNumArguments() != functionType.getNumInputs()) {
    return emitOpError("body argument count does not match the gate signature");
  }
  for (const auto [argument, expected] :
       llvm::zip_equal(entry.getArgumentTypes(), functionType.getInputs())) {
    if (argument != expected) {
      return emitOpError("body argument types do not match the gate signature");
    }
  }
  if (functionType.getNumResults() != 0) {
    return emitOpError("gate definitions cannot return classical values");
  }
  auto bodyIsUnitary = getBody().walk([&](Operation* operation) {
    if (isa<ApplyGateOp>(operation)) {
      return WalkResult::advance();
    }
    const auto hasQubitType = [](Type type) {
      return isa<qc::QubitType>(type);
    };
    if (llvm::any_of(operation->getOperandTypes(), hasQubitType) ||
        llvm::any_of(operation->getResultTypes(), hasQubitType)) {
      return WalkResult::interrupt();
    }
    if (isa<YieldOp, cf::AssertOp>(operation) ||
        isMemoryEffectFree(operation) ||
        operation->getName().getDialectNamespace() == "scf") {
      return WalkResult::advance();
    }
    return WalkResult::interrupt();
  });
  if (bodyIsUnitary.wasInterrupted()) {
    return emitOpError(
        "gate bodies may contain only pure parameter computation and gate "
        "applications");
  }
  return verifyGateSignature(getOperation(), functionType);
}

LogicalResult GateDeclOp::verify() {
  auto type = dyn_cast<FunctionType>(getFunctionType());
  if (!type) {
    return emitOpError("requires a function type");
  }
  if (type.getNumResults() != 0) {
    return emitOpError("gate declarations cannot return values");
  }
  if (const auto* catalog = lookupGate(getSymName())) {
    if (type.getNumInputs() !=
            catalog->parameterCount + catalog->qubitCount() ||
        llvm::any_of(type.getInputs().take_front(catalog->parameterCount),
                     [](Type input) { return !input.isF64(); }) ||
        llvm::any_of(type.getInputs().drop_front(catalog->parameterCount),
                     [](Type input) { return !isa<qc::QubitType>(input); })) {
      return emitOpError(
          "catalog gate signature does not match its canonical declaration");
    }
  }
  return verifyGateSignature(getOperation(), type);
}

static bool insertKnownPhysicalQubit(
    Value qubit, llvm::DenseSet<std::uint64_t>& staticIndices,
    llvm::DenseMap<Value, llvm::DenseSet<Value>>& dynamicLoadIndices,
    llvm::DenseMap<Value, llvm::DenseSet<APInt>>& constantLoadIndices) {
  if (auto staticQubit = qubit.getDefiningOp<qc::StaticOp>()) {
    return staticIndices.insert(staticQubit.getIndex()).second;
  }

  auto load = qubit.getDefiningOp<memref::LoadOp>();
  if (!load || load.getIndices().size() != 1) {
    return true;
  }
  const auto memory = load.getMemRef();
  const auto index = load.getIndices().front();
  APInt constantIndex;
  if (matchPattern(index, m_ConstantInt(&constantIndex))) {
    return constantLoadIndices[memory].insert(constantIndex).second;
  }
  return dynamicLoadIndices[memory].insert(index).second;
}

LogicalResult ApplyGateOp::verify() {
  Operation* declaration =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getCalleeAttr());
  if (declaration == nullptr || !isa<GateOp, GateDeclOp>(declaration)) {
    return emitOpError("references an unknown gate symbol '")
           << getCallee() << "'";
  }
  auto declaredType = isa<GateOp>(declaration)
                          ? cast<GateOp>(declaration).getFunctionType()
                          : cast<GateDeclOp>(declaration).getFunctionType();
  auto functionType = dyn_cast<FunctionType>(declaredType);
  if (!functionType) {
    return emitOpError("references a gate without a function signature");
  }
  const auto firstQubit =
      llvm::find_if(functionType.getInputs(),
                    [](Type type) { return isa<qc::QubitType>(type); });
  const size_t parameterCount =
      std::distance(functionType.getInputs().begin(), firstQubit);
  const size_t baseQubitCount = functionType.getNumInputs() - parameterCount;
  if (getParameters().size() != parameterCount ||
      !llvm::equal(getParameters().getTypes(),
                   functionType.getInputs().take_front(parameterCount))) {
    return emitOpError(
        "operand types do not match the referenced gate signature");
  }
  llvm::DenseSet<Value> distinctQubits;
  llvm::DenseSet<std::uint64_t> staticIndices;
  llvm::DenseMap<Value, llvm::DenseSet<Value>> dynamicLoadIndices;
  llvm::DenseMap<Value, llvm::DenseSet<APInt>> constantLoadIndices;
  for (auto qubit : getQubits()) {
    if (!distinctQubits.insert(qubit).second) {
      return emitOpError("qubit operands must be distinct");
    }
    // The verifier rejects aliases that canonical QC producers make
    // decidable. Distinct dynamic indices remain legal because arbitrary
    // runtime equality cannot be proven here; producers must guard those
    // applications when aliasing is possible.
    if (!insertKnownPhysicalQubit(qubit, staticIndices, dynamicLoadIndices,
                                  constantLoadIndices)) {
      return emitOpError("qubit operands are known to physically alias");
    }
  }
  if (baseQubitCount > getQubits().size()) {
    return emitOpError("qubit operands do not match the referenced gate "
                       "signature");
  }
  const size_t availableControlCount = getQubits().size() - baseQubitCount;

  const auto kinds = getModifierKinds();
  const auto indices = getModifierOperandIndices();
  if (kinds.size() != indices.size()) {
    return emitOpError("requires one operand index per gate modifier");
  }

  llvm::SmallBitVector used(getModifierOperands().size());
  size_t knownControlCount = 0;
  for (const auto [position, rawKind] : llvm::enumerate(kinds)) {
    if (rawKind < static_cast<int32_t>(GateModifierKind::inv) ||
        rawKind > static_cast<int32_t>(GateModifierKind::pow)) {
      return emitOpError("contains an unknown gate modifier kind");
    }
    const auto kind = static_cast<GateModifierKind>(rawKind);
    const bool isControl =
        kind == GateModifierKind::ctrl || kind == GateModifierKind::negctrl;
    const int32_t index = indices[position];
    const bool permitsOperand = kind == GateModifierKind::pow ||
                                kind == GateModifierKind::ctrl ||
                                kind == GateModifierKind::negctrl;
    const bool requiresOperand = kind == GateModifierKind::pow;
    if (!permitsOperand && index != -1) {
      return emitOpError("inv modifiers cannot reference an operand");
    }
    if (requiresOperand && index < 0) {
      return emitOpError("pow modifiers require an exponent operand");
    }
    if (index >= 0) {
      if (static_cast<size_t>(index) >= getModifierOperands().size()) {
        return emitOpError("modifier operand index is out of bounds");
      }
      if (used.test(index)) {
        return emitOpError("modifier operands must be referenced exactly once");
      }
      used.set(index);
      auto operandType = getModifierOperands()[index].getType();
      if ((kind == GateModifierKind::ctrl ||
           kind == GateModifierKind::negctrl) &&
          !isa<IntegerType>(operandType)) {
        return emitOpError("control modifier operands must have an integer "
                           "type");
      }
      if (isControl) {
        if (auto constant = getModifierOperands()[index]
                                .getDefiningOp<arith::ConstantIntOp>()) {
          if (constant.value() <= 0) {
            return emitOpError("control counts must be positive");
          }
          const auto count = static_cast<size_t>(constant.value());
          if (count > availableControlCount - knownControlCount) {
            return emitOpError("qubit operands do not match the referenced "
                               "gate signature");
          }
          knownControlCount += count;
        } else {
          return emitOpError("control counts must be constant integers");
        }
      }
    } else if (isControl) {
      if (knownControlCount == availableControlCount) {
        return emitOpError("qubit operands do not match the referenced gate "
                           "signature");
      }
      ++knownControlCount;
    }
  }
  if (used.count() != getModifierOperands().size()) {
    return emitOpError("contains an unreferenced modifier operand");
  }
  if (knownControlCount != availableControlCount) {
    return emitOpError("qubit operands do not match the referenced gate "
                       "signature");
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OQ3/IR/OQ3Ops.cpp.inc"
