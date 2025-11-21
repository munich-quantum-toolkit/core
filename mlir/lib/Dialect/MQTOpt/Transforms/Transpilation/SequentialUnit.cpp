/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <stdexcept>
#include <utility>

namespace mqt::ir::opt {

SequentialUnit::SequentialUnit(Layout layout, mlir::Region* region,
                               mlir::Region::OpIterator start, bool restore)
    : Unit(std::move(layout), region, restore), start_(start),
      end_(region->op_end()) {
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

[[nodiscard]] mlir::SmallVector<SequentialUnit, 3> SequentialUnit::next() {
  if (divider_ == nullptr) {
    return {};
  }

  mlir::SmallVector<SequentialUnit, 3> units;
  mlir::TypeSwitch<mlir::Operation*>(divider_)
      .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
        /// Copy layout.
        Layout forLayout(layout_);

        /// Forward out-of-loop and in-loop values.
        const auto nqubits = layout_.getNumQubits();
        const auto initArgs = op.getInitArgs().take_front(nqubits);
        const auto results = op.getResults().take_front(nqubits);
        const auto iterArgs = op.getRegionIterArgs().take_front(nqubits);
        for (const auto [arg, res, iter] :
             llvm::zip(initArgs, results, iterArgs)) {
          layout_.remapQubitValue(arg, res);
          forLayout.remapQubitValue(arg, iter);
        }

        units.emplace_back(std::move(layout_), region_, std::next(end_),
                           restore_);
        units.emplace_back(std::move(forLayout), &op.getRegion(), true);
      })
      .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp op) {
        units.emplace_back(layout_, &op.getThenRegion(), true);
        units.emplace_back(layout_, &op.getElseRegion(), true);

        /// Forward results.
        const auto results =
            op->getResults().take_front(layout_.getNumQubits());
        for (const auto [in, out] :
             llvm::zip(layout_.getHardwareQubits(), results)) {
          layout_.remapQubitValue(in, out);
        }

        units.emplace_back(std::move(layout_), region_, std::next(end_),
                           restore_);
      })
      .Default(
          [](auto) { throw std::runtime_error("invalid 'next' operation"); });

  return units;
}
} // namespace mqt::ir::opt
