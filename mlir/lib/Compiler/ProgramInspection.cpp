/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

namespace mlir {
static QuantumProgramInfo inspectProgram(ModuleOp moduleOp) {
  QuantumProgramInfo info;
  auto entry = mqt::getEntryPoint(moduleOp);
  if (entry && llvm::none_of(entry.getArgumentTypes(), [](Type type) {
        return isa<qc::QubitType, qco::QubitType>(type) ||
               (isa<ShapedType>(type) &&
                isa<qc::QubitType, qco::QubitType>(
                    cast<ShapedType>(type).getElementType()));
      })) {
    info.numQubits = 0;
  }
  const auto addAllocation = [&](const uint64_t size) {
    if (!info.numQubits) {
      return;
    }
    if (size > std::numeric_limits<uint64_t>::max() - *info.numQubits) {
      info.numQubits.reset();
    } else {
      *info.numQubits += size;
    }
  };
  moduleOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<ModuleOp>(op) && op != moduleOp) {
      return WalkResult::skip();
    }
    info.hasControlFlow |= isa<BranchOpInterface, RegionBranchOpInterface>(op);
    if (auto qubit = dyn_cast<qc::StaticOp>(op)) {
      info.staticQubits.push_back(qubit.getIndex());
    } else if (auto qubit = dyn_cast<qco::StaticOp>(op)) {
      info.staticQubits.push_back(qubit.getIndex());
    } else if (isa<qc::AllocOp, qco::AllocOp>(op)) {
      addAllocation(1);
    } else {
      ShapedType type;
      if (auto alloc = dyn_cast<memref::AllocOp>(op);
          alloc && isa<qc::QubitType>(alloc.getType().getElementType())) {
        type = alloc.getType();
      } else if (auto alloc = dyn_cast<qtensor::AllocOp>(op)) {
        type = alloc.getType();
      }
      if (type) {
        if (type.hasStaticShape()) {
          addAllocation(static_cast<uint64_t>(type.getNumElements()));
        } else {
          info.numQubits.reset();
        }
      }
    }
    return WalkResult::advance();
  });
  llvm::sort(info.staticQubits);
  const auto duplicates = std::ranges::unique(info.staticQubits);
  info.staticQubits.erase(duplicates.begin(), duplicates.end());
  if (info.numQubits && !info.staticQubits.empty()) {
    info.numQubits = info.staticQubits.size();
  }
  return info;
}
QuantumProgramInfo QCProgram::inspect() const { return inspectProgram(mod()); }
QuantumProgramInfo QCOProgram::inspect() const { return inspectProgram(mod()); }

static void forEachGate(ModuleOp moduleOp,
                        function_ref<void(qc::UnitaryOpInterface)> visit) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return;
  }
  entryPoint.walk<WalkOrder::PreOrder>([&](qc::UnitaryOpInterface op) {
    if (!isa<qc::BarrierOp>(op)) {
      visit(op);
    }
    return isa<qc::CtrlOp, qc::InvOp, qc::PowOp>(op) ? WalkResult::skip()
                                                     : WalkResult::advance();
  });
}

size_t QCProgram::numGates() const {
  size_t count = 0;
  forEachGate(mod(), [&](qc::UnitaryOpInterface) { ++count; });
  return count;
}

size_t QCProgram::numSingleQubitGates() const {
  size_t count = 0;
  forEachGate(mod(),
              [&](qc::UnitaryOpInterface op) { count += op.isSingleQubit(); });
  return count;
}

size_t QCProgram::numTwoQubitGates() const {
  size_t count = 0;
  forEachGate(mod(),
              [&](qc::UnitaryOpInterface op) { count += op.isTwoQubit(); });
  return count;
}

std::map<std::string, size_t> QCProgram::gateCounts() const {
  std::map<std::string, size_t> counts;
  forEachGate(mod(), [&](qc::UnitaryOpInterface op) {
    ++counts[op.getBaseSymbol().str()];
  });
  return counts;
}
} // namespace mlir
