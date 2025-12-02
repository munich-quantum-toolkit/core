/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::opt {
namespace {
/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr mlir::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

/**
 * @brief Attribute to forward function-level attributes to LLVM IR.
 */
constexpr mlir::StringLiteral PASSTHROUGH_ATTR{"passthrough"};
} // namespace

bool isEntryPoint(mlir::func::FuncOp op) {
  const auto passthroughAttr =
      op->getAttrOfType<mlir::ArrayAttr>(PASSTHROUGH_ATTR);
  if (!passthroughAttr) {
    return false;
  }

  return llvm::any_of(passthroughAttr, [](const mlir::Attribute attr) {
    return mlir::isa<mlir::StringAttr>(attr) &&
           mlir::cast<mlir::StringAttr>(attr) == ENTRY_POINT_ATTR;
  });
}

bool isTwoQubitGate(UnitaryInterface op) {
  return (op.getInQubits().size() + op.getPosCtrlInQubits().size() +
          op.getNegCtrlInQubits().size()) == 2;
}

[[nodiscard]] ValuePair getIns(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const auto target = op.getInQubits();
  const auto targetSize = target.size();

  if (targetSize == 2) {
    return {target[0], target[1]};
  }

  const auto posCtrl = op.getPosCtrlInQubits();
  return (posCtrl.size() == 1)
             ? std::pair{target[0], posCtrl[0]}
             : std::pair{target[0], op.getNegCtrlInQubits()[0]};
}

[[nodiscard]] ValuePair getOuts(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const auto target = op.getOutQubits();
  const auto targetSize = target.size();

  if (targetSize == 2) {
    return {target[0], target[1]};
  }

  const auto posCtrl = op.getPosCtrlOutQubits();
  return (posCtrl.size() == 1)
             ? std::pair{target[0], posCtrl[0]}
             : std::pair{target[0], op.getNegCtrlOutQubits()[0]};
}

[[nodiscard]] mlir::Operation* getUserInRegion(mlir::Value v,
                                               mlir::Region* region) {
  for (mlir::Operation* user : v.getUsers()) {
    if (user->getParentRegion() == region) {
      return user;
    }
  }
  return nullptr;
}

[[nodiscard]] SWAPOp createSwap(mlir::Location location, mlir::Value in0,
                                mlir::Value in1,
                                mlir::PatternRewriter& rewriter) {
  const mlir::SmallVector<mlir::Type> resultTypes{in0.getType(), in1.getType()};
  const mlir::SmallVector<mlir::Value> inQubits{in0, in1};

  return rewriter.create<SWAPOp>(
      /* location = */ location,
      /* out_qubits = */ resultTypes,
      /* pos_ctrl_out_qubits = */ mlir::TypeRange{},
      /* neg_ctrl_out_qubits = */ mlir::TypeRange{},
      /* static_params = */ nullptr,
      /* params_mask = */ nullptr,
      /* params = */ mlir::ValueRange{},
      /* in_qubits = */ inQubits,
      /* pos_ctrl_in_qubits = */ mlir::ValueRange{},
      /* neg_ctrl_in_qubits = */ mlir::ValueRange{});
}

void replaceAllUsesInRegionAndChildrenExcept(mlir::Value oldValue,
                                             mlir::Value newValue,
                                             mlir::Region* region,
                                             mlir::Operation* exceptOp,
                                             mlir::PatternRewriter& rewriter) {
  if (oldValue == newValue) {
    return;
  }

  rewriter.replaceUsesWithIf(oldValue, newValue, [&](mlir::OpOperand& use) {
    mlir::Operation* user = use.getOwner();
    if (user == exceptOp) {
      return false;
    }

    // For other blocks, check if in region tree
    mlir::Region* userRegion = user->getParentRegion();
    while (userRegion) {
      if (userRegion == region) {
        return true;
      }
      userRegion = userRegion->getParentRegion();
    }
    return false;
  });
}

[[nodiscard]] bool isExecutable(UnitaryInterface op, const Layout& layout,
                                const Architecture& arch) {
  assert(isTwoQubitGate(op));
  const auto ins = getIns(op);
  return arch.areAdjacent(layout.lookupHardwareIndex(ins.first),
                          layout.lookupHardwareIndex(ins.second));
}

void insertSWAPs(mlir::Location loc, mlir::ArrayRef<QubitIndexPair> swaps,
                 Layout& layout, mlir::PatternRewriter& rewriter) {
  for (const auto [hw0, hw1] : swaps) {
    const mlir::Value in0 = layout.lookupHardwareValue(hw0);
    const mlir::Value in1 = layout.lookupHardwareValue(hw1);
    [[maybe_unused]] const auto [prog0, prog1] =
        layout.getProgramIndices(hw0, hw1);

    auto swap = createSwap(loc, in0, in1, rewriter);
    const auto [out0, out1] = getOuts(swap);

    rewriter.setInsertionPointAfter(swap);
    replaceAllUsesInRegionAndChildrenExcept(in0, out1, swap->getParentRegion(),
                                            swap, rewriter);
    replaceAllUsesInRegionAndChildrenExcept(in1, out0, swap->getParentRegion(),
                                            swap, rewriter);

    layout.swap(in0, in1);
    layout.remapQubitValue(in0, out0);
    layout.remapQubitValue(in1, out1);
  }
}
} // namespace mqt::ir::opt
