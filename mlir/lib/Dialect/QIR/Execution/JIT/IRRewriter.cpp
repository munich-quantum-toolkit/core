/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Execution/JIT/IRRewriter.h"

#include "mlir/Dialect/QIR/QIRDefinitions.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <llvm/Transforms/Utils/Local.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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
      if (callee != nullptr &&
          callee->getName().starts_with("__quantum__qis__") &&
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

static std::optional<uintptr_t> staticResourceId(const llvm::Value* value) {
  if (llvm::isa<llvm::ConstantPointerNull>(value)) {
    return 0;
  }
  const auto* cast = llvm::dyn_cast<llvm::ConstantExpr>(value);
  if (cast == nullptr || cast->getOpcode() != llvm::Instruction::IntToPtr) {
    return std::nullopt;
  }
  const auto* id = llvm::dyn_cast<llvm::ConstantInt>(cast->getOperand(0));
  if (id == nullptr ||
      id->getValue().getActiveBits() > std::numeric_limits<uintptr_t>::digits) {
    return std::nullopt;
  }
  return static_cast<uintptr_t>(id->getZExtValue());
}

static bool isStaticGate(const llvm::CallInst& call, llvm::StringRef name) {
  const auto known = llvm::StringSwitch<bool>(name)
#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  .Case("__quantum__qis__" #NAME "__" #SUFFIX, true)                           \
      .Case("__quantum__qis__c" #NAME "__" #SUFFIX, true)                      \
      .Case("__quantum__qis__cc" #NAME "__" #SUFFIX, true)
#include "mlir/Conversion/GateTable.def"
                         .Case("__quantum__qis__gphase__body", true)
                         .Case("__quantum__qis__cnot__body", true)
                         .Default(false);
  return known && std::ranges::all_of(call.args(), [](const llvm::Value* arg) {
           if (const auto* fp = llvm::dyn_cast<llvm::ConstantFP>(arg)) {
             return fp->getValueAPF().isFinite();
           }
           return staticResourceId(arg).has_value();
         });
}

std::optional<std::vector<uintptr_t>>
getStaticSamplingOutputs(const llvm::Function& entryPoint) {
  const auto profile = entryPoint.getFnAttribute(QIR_PROFILES_ATTR);
  if (entryPoint.isDeclaration() || !profile.isStringAttribute() ||
      profile.getValueAsString().compare(BASE_PROFILE) != 0) {
    return std::nullopt;
  }
  std::unordered_map<uintptr_t, uintptr_t> results;
  std::vector<uintptr_t> outputs;
  llvm::SmallPtrSet<const llvm::BasicBlock*, 4> visited;
  const auto* block = &entryPoint.getEntryBlock();
  bool terminal = false;
  bool started = false;
  while (visited.insert(block).second) {
    for (const auto& instruction : *block) {
      if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
        const auto* code =
            llvm::dyn_cast_or_null<llvm::ConstantInt>(ret->getReturnValue());
        if (code != nullptr && code->isZero()) {
          return outputs;
        }
        return std::nullopt;
      }
      if (const auto* branch = llvm::dyn_cast<llvm::BranchInst>(&instruction)) {
        if (branch->isConditional()) {
          return std::nullopt;
        }
        break;
      }
      const auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction);
      const auto* callee =
          call == nullptr ? nullptr : call->getCalledFunction();
      if (callee == nullptr || !callee->isDeclaration()) {
        return std::nullopt;
      }
      const auto name = callee->getName();
      if (name == "__quantum__rt__initialize" && !started &&
          call->arg_size() == 1 &&
          llvm::isa<llvm::ConstantPointerNull>(call->getArgOperand(0))) {
        started = true;
        continue;
      }
      started = true;
      if (!terminal && isStaticGate(*call, name)) {
        continue;
      }
      if (name == "__quantum__qis__mz__body" && call->arg_size() == 2) {
        const auto qubit = staticResourceId(call->getArgOperand(0));
        const auto result = staticResourceId(call->getArgOperand(1));
        if (!qubit || !result) {
          return std::nullopt;
        }
        terminal = true;
        results[*result] = *qubit;
        continue;
      }
      if (name == "__quantum__rt__result_record_output" &&
          call->arg_size() == 2) {
        const auto result = staticResourceId(call->getArgOperand(0));
        if (!result) {
          return std::nullopt;
        }
        const auto it = results.find(*result);
        if (it == results.end()) {
          return std::nullopt;
        }
        terminal = true;
        outputs.push_back(it->second);
        continue;
      }
      if ((name == "__quantum__rt__tuple_record_output" ||
           name == "__quantum__rt__array_record_output" ||
           name == "__quantum__rt__bool_record_output" ||
           name == "__quantum__rt__int_record_output" ||
           name == "__quantum__rt__double_record_output") &&
          call->arg_size() == 2 &&
          (llvm::isa<llvm::ConstantInt>(call->getArgOperand(0)) ||
           llvm::isa<llvm::ConstantFP>(call->getArgOperand(0)))) {
        terminal = true;
        continue;
      }
      return std::nullopt;
    }
    const auto* branch =
        llvm::dyn_cast<llvm::BranchInst>(block->getTerminator());
    if (branch == nullptr || branch->isConditional()) {
      return std::nullopt;
    }
    block = branch->getSuccessor(0);
  }
  return std::nullopt;
}

} // namespace qir
