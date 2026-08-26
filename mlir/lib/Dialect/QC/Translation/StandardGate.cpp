/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/StandardGate.h"

#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cstddef>

namespace mlir::qc {
namespace {

constexpr std::array DESCRIPTORS{
    StandardGateDescriptor{StandardGate::GPhase, "gphase", 1, 0, 0},
    StandardGateDescriptor{StandardGate::Id, "id", 0, 0, 1},
    StandardGateDescriptor{StandardGate::X, "x", 0, 0, 1},
    StandardGateDescriptor{StandardGate::Y, "y", 0, 0, 1},
    StandardGateDescriptor{StandardGate::Z, "z", 0, 0, 1},
    StandardGateDescriptor{StandardGate::H, "h", 0, 0, 1},
    StandardGateDescriptor{StandardGate::S, "s", 0, 0, 1},
    StandardGateDescriptor{StandardGate::Sdg, "sdg", 0, 0, 1},
    StandardGateDescriptor{StandardGate::T, "t", 0, 0, 1},
    StandardGateDescriptor{StandardGate::Tdg, "tdg", 0, 0, 1},
    StandardGateDescriptor{StandardGate::SX, "sx", 0, 0, 1},
    StandardGateDescriptor{StandardGate::SXdg, "sxdg", 0, 0, 1},
    StandardGateDescriptor{StandardGate::P, "p", 1, 0, 1},
    StandardGateDescriptor{StandardGate::RX, "rx", 1, 0, 1},
    StandardGateDescriptor{StandardGate::RY, "ry", 1, 0, 1},
    StandardGateDescriptor{StandardGate::RZ, "rz", 1, 0, 1},
    StandardGateDescriptor{StandardGate::R, "r", 2, 0, 1},
    StandardGateDescriptor{StandardGate::U2, "u2", 2, 0, 1},
    StandardGateDescriptor{StandardGate::U3, "u", 3, 0, 1},
    StandardGateDescriptor{StandardGate::BuiltinU, "U", 3, 0, 1},
    StandardGateDescriptor{StandardGate::CU, "cu", 4, 1, 1},
    StandardGateDescriptor{StandardGate::SWAP, "swap", 0, 0, 2},
    StandardGateDescriptor{StandardGate::ISWAP, "iswap", 0, 0, 2},
    StandardGateDescriptor{StandardGate::DCX, "dcx", 0, 0, 2},
    StandardGateDescriptor{StandardGate::ECR, "ecr", 0, 0, 2},
    StandardGateDescriptor{StandardGate::RCCX, "rccx", 0, 0, 3},
    StandardGateDescriptor{StandardGate::RXX, "rxx", 1, 0, 2},
    StandardGateDescriptor{StandardGate::RYY, "ryy", 1, 0, 2},
    StandardGateDescriptor{StandardGate::RZX, "rzx", 1, 0, 2},
    StandardGateDescriptor{StandardGate::RZZ, "rzz", 1, 0, 2},
    StandardGateDescriptor{StandardGate::XXPlusYY, "xx_plus_yy", 2, 0, 2},
    StandardGateDescriptor{StandardGate::XXMinusYY, "xx_minus_yy", 2, 0, 2},
};

static_assert(
    [] {
      for (size_t index = 0; index < DESCRIPTORS.size(); ++index) {
        if (static_cast<size_t>(DESCRIPTORS[index].gate) != index) {
          return false;
        }
      }
      return true;
    }(),
    "standard-gate descriptors must follow enum order");

} // namespace

const StandardGateDescriptor&
getStandardGateDescriptor(const StandardGate gate) {
  return DESCRIPTORS.at(static_cast<size_t>(gate));
}

const StandardGateDescriptor*
lookupStandardGateByOperationSymbol(const llvm::StringRef symbol) {
  for (const auto& descriptor : DESCRIPTORS) {
    if (descriptor.operationSymbol == symbol) {
      return &descriptor;
    }
  }
  return nullptr;
}

LogicalResult emitStandardGate(OpBuilder& builder, const Location loc,
                               const StandardGate gate, ValueRange parameters,
                               ValueRange qubits) {
  const auto& descriptor = getStandardGateDescriptor(gate);
  if (parameters.size() != descriptor.parameterCount ||
      qubits.size() != descriptor.targetCount) {
    return failure();
  }

  llvm::StringRef operationName;
  switch (gate) {
  case StandardGate::GPhase:
    operationName = GPhaseOp::getOperationName();
    break;
  case StandardGate::Id:
    operationName = IdOp::getOperationName();
    break;
  case StandardGate::X:
    operationName = XOp::getOperationName();
    break;
  case StandardGate::Y:
    operationName = YOp::getOperationName();
    break;
  case StandardGate::Z:
    operationName = ZOp::getOperationName();
    break;
  case StandardGate::H:
    operationName = HOp::getOperationName();
    break;
  case StandardGate::S:
    operationName = SOp::getOperationName();
    break;
  case StandardGate::Sdg:
    operationName = SdgOp::getOperationName();
    break;
  case StandardGate::T:
    operationName = TOp::getOperationName();
    break;
  case StandardGate::Tdg:
    operationName = TdgOp::getOperationName();
    break;
  case StandardGate::SX:
    operationName = SXOp::getOperationName();
    break;
  case StandardGate::SXdg:
    operationName = SXdgOp::getOperationName();
    break;
  case StandardGate::P:
    operationName = POp::getOperationName();
    break;
  case StandardGate::RX:
    operationName = RXOp::getOperationName();
    break;
  case StandardGate::RY:
    operationName = RYOp::getOperationName();
    break;
  case StandardGate::RZ:
    operationName = RZOp::getOperationName();
    break;
  case StandardGate::R:
    operationName = ROp::getOperationName();
    break;
  case StandardGate::U2:
    operationName = U2Op::getOperationName();
    break;
  case StandardGate::U3:
    operationName = UOp::getOperationName();
    break;
  case StandardGate::SWAP:
    operationName = SWAPOp::getOperationName();
    break;
  case StandardGate::ISWAP:
    operationName = iSWAPOp::getOperationName();
    break;
  case StandardGate::DCX:
    operationName = DCXOp::getOperationName();
    break;
  case StandardGate::ECR:
    operationName = ECROp::getOperationName();
    break;
  case StandardGate::RCCX:
    operationName = RCCXOp::getOperationName();
    break;
  case StandardGate::RXX:
    operationName = RXXOp::getOperationName();
    break;
  case StandardGate::RYY:
    operationName = RYYOp::getOperationName();
    break;
  case StandardGate::RZX:
    operationName = RZXOp::getOperationName();
    break;
  case StandardGate::RZZ:
    operationName = RZZOp::getOperationName();
    break;
  case StandardGate::XXPlusYY:
    operationName = XXPlusYYOp::getOperationName();
    break;
  case StandardGate::XXMinusYY:
    operationName = XXMinusYYOp::getOperationName();
    break;
  case StandardGate::BuiltinU:
  case StandardGate::CU:
    return failure();
  }
  OperationState state(loc, operationName);
  state.addOperands(gate == StandardGate::GPhase ? parameters : qubits);
  if (gate != StandardGate::GPhase) {
    state.addOperands(parameters);
  }
  builder.create(state);
  return success();
}

} // namespace mlir::qc
