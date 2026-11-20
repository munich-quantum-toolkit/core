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

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/FunctionUtils.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>
#include <optional>

namespace mlir::qco {

/// Trace physical sites through positional wires. Attach the listener when
/// rewriting IR so changes invalidate cached paths.
class StaticSites final : public RewriterBase::Listener {
public:
  std::optional<CompilerTarget::SiteId> lookup(Value value) {
    SmallVector<Value> path;
    const auto site = [&]() -> std::optional<CompilerTarget::SiteId> {
      while (true) {
        if (const auto known = sites_.find(value); known != sites_.end()) {
          return known->second;
        }
        path.push_back(value);
        if (auto site = value.getDefiningOp<StaticOp>()) {
          return site.getIndex();
        }
        // Entry sites follow verified positional wires; final target
        // conformance still checks every branch join and loop backedge.
        if (auto argument = dyn_cast<BlockArgument>(value)) {
          auto* parent = argument.getOwner()->getParentOp();
          if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            value = loop.getTiedLoopInit(argument)->get();
          } else if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
            auto inputs = argument.getOwner() == loop.getBeforeBody()
                              ? loop.getInits()
                              : loop.getConditionOp().getArgs();
            value = inputs[argument.getArgNumber()];
          } else if (auto branch = dyn_cast<IfOp>(parent)) {
            value = branch.getQubits()[argument.getArgNumber()];
          } else if (auto branch = dyn_cast<IndexSwitchOp>(parent)) {
            value = branch.getTargets()[argument.getArgNumber()];
          } else {
            return std::nullopt;
          }
          continue;
        }
        if (auto loop = value.getDefiningOp<scf::WhileOp>()) {
          const auto result = cast<OpResult>(value).getResultNumber();
          auto argument = traceQubitArgument(
              *loop.getBeforeBody(), loop.getConditionOp().getArgs()[result]);
          if (failed(argument)) {
            return std::nullopt;
          }
          value = loop.getInits()[*argument];
          continue;
        }
        WireIterator iterator(value);
        --iterator;
        if (iterator == std::default_sentinel) {
          return std::nullopt;
        }
        value = iterator.qubit();
      }
    }();
    if (site) {
      for (Value visited : path) {
        sites_.try_emplace(visited, *site);
      }
    }
    return site;
  }

  void notifyOperationModified(Operation*) override { sites_.clear(); }
  void notifyOperationErased(Operation*) override { sites_.clear(); }
  void notifyBlockErased(Block*) override { sites_.clear(); }

private:
  DenseMap<Value, CompilerTarget::SiteId> sites_;
};

} // namespace mlir::qco
