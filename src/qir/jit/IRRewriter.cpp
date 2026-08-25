/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/jit/IRRewriter.hpp"

#include "qir/Definitions.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <llvm/Transforms/Utils/Local.h>

#include <stdexcept>

namespace qir {

static constexpr llvm::StringLiteral TERMINAL_REGION_ERROR =
    "QIR state extraction requires irreversible operations to form a terminal "
    "region";

static bool isIrreversible(const llvm::CallBase& call) {
  const auto* callee = llvm::dyn_cast<llvm::Function>(
      call.getCalledOperand()->stripPointerCasts());
  return callee != nullptr && callee->hasFnAttribute(IRREVERSIBLE_ATTR);
}

static void requireTerminalIrreversibleRegion(llvm::CallInst& boundary) {
  llvm::SmallPtrSet<llvm::BasicBlock*, 8> visited;
  llvm::SmallVector<llvm::BasicBlock*, 8> pending;

  auto inspect = [&](llvm::BasicBlock& block,
                     llvm::BasicBlock::iterator begin) {
    for (auto it = begin; it != block.end(); ++it) {
      const auto* call = llvm::dyn_cast<llvm::CallBase>(&*it);
      const auto* callee =
          call == nullptr ? nullptr
                          : llvm::dyn_cast<llvm::Function>(
                                call->getCalledOperand()->stripPointerCasts());
      if (callee != nullptr && callee->getName().starts_with(QIS_PREFIX) &&
          !isIrreversible(*call)) {
        throw std::invalid_argument(TERMINAL_REGION_ERROR.str());
      }
    }

    auto* terminator = block.getTerminator();
    if (terminator->getNumSuccessors() > 1) {
      throw std::invalid_argument(TERMINAL_REGION_ERROR.str());
    }
    for (auto* successor : llvm::successors(&block)) {
      pending.emplace_back(successor);
    }
  };

  auto* boundaryBlock = boundary.getParent();
  visited.insert(boundaryBlock);
  inspect(*boundaryBlock, boundary.getIterator());
  while (!pending.empty()) {
    auto* block = pending.pop_back_val();
    if (!visited.insert(block).second) {
      throw std::invalid_argument(TERMINAL_REGION_ERROR.str());
    }
    inspect(*block, block->begin());
  }
}

bool prepareForStateExtraction(llvm::Function& entryPoint) {
  if (!entryPoint.getReturnType()->isIntegerTy(64) || !entryPoint.arg_empty()) {
    throw std::invalid_argument(
        "QIR state extraction requires an i64() entry point");
  }

  const auto profile = entryPoint.getFnAttribute(QIR_PROFILES_ATTR);
  if (!profile.isStringAttribute() ||
      profile.getValueAsString().compare(BASE_PROFILE) != 0) {
    throw std::invalid_argument(
        "QIR state extraction requires a Base Profile entry point");
  }

  llvm::SmallVector<llvm::CallInst*, 8> irreversibleCalls;
  for (auto& block : entryPoint) {
    for (auto& instruction : block) {
      auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction);
      if (call != nullptr && isIrreversible(*call)) {
        irreversibleCalls.emplace_back(call);
      }
    }
  }
  if (irreversibleCalls.empty()) {
    return false;
  }

  const llvm::DominatorTree dominators(entryPoint);
  llvm::CallInst* boundary = nullptr;
  for (auto* candidate : irreversibleCalls) {
    bool dominatesAll = true;
    for (auto* call : irreversibleCalls) {
      if (candidate != call && !dominators.dominates(candidate, call)) {
        dominatesAll = false;
        break;
      }
    }
    if (dominatesAll) {
      boundary = candidate;
      break;
    }
  }
  if (boundary == nullptr) {
    throw std::invalid_argument(TERMINAL_REGION_ERROR.str());
  }

  requireTerminalIrreversibleRegion(*boundary);
  auto* prefix = boundary->getParent();
  prefix->splitBasicBlock(boundary, "state-extraction.discarded");
  auto* oldTerminator = prefix->getTerminator();
  llvm::IRBuilder<> builder(oldTerminator);
  builder.CreateRet(llvm::Constant::getNullValue(entryPoint.getReturnType()));
  oldTerminator->eraseFromParent();
  llvm::removeUnreachableBlocks(entryPoint);
  return true;
}

} // namespace qir
