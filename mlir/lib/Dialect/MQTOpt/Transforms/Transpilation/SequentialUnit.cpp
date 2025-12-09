/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/SequentialUnit.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::opt {

SequentialUnit
SequentialUnit::fromEntryPointFunction(mlir::func::FuncOp func,
                                       const std::size_t nqubits) {
  Layout layout(nqubits);
  for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
    layout.add(op.getIndex(), op.getIndex(), op.getQubit());
  });
  return {std::move(layout), &func.getBody()};
}

SequentialUnit::SequentialUnit(Layout layout, mlir::Region* region,
                               mlir::Region::OpIterator start)
    : Unit(std::move(layout), region), start_(start), end_(region->op_end()) {
  mlir::Region::OpIterator it = start_;
  for (; it != end_; ++it) {
    mlir::Operation* op = &*it;
    if (mlir::isa<mlir::RegionBranchOpInterface>(op)) {
      divider_ = op;
      break;
    }
  }
  end_ = it;
}

mlir::SmallVector<SequentialUnit, 3> SequentialUnit::nextImpl() {
  if (divider_ == nullptr) {
    return {};
  }

  mlir::SmallVector<SequentialUnit, 3> units;
  mlir::TypeSwitch<mlir::Operation*>(divider_)
      .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
        Layout forLayout(layout_); // Copy layout.
        forLayout.remapToLoopBody(op);
        layout_.remapToLoopResults(op);
        units.emplace_back(std::move(layout_), region_, std::next(end_));
        units.emplace_back(std::move(forLayout), &op.getRegion());
      })
      .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
        units.emplace_back(layout_, &op.getThenRegion());
        units.emplace_back(layout_, &op.getElseRegion());
        layout_.remapIfResults(op);
        units.emplace_back(std::move(layout_), region_, std::next(end_));
      })
      .Default([](auto) { llvm_unreachable("invalid 'next' operation"); });

  return units;
}
} // namespace mqt::ir::opt
