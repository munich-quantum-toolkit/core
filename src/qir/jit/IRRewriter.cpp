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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <array>
#include <cstddef>

namespace qir {

static constexpr std::array<llvm::StringRef, 5> STRIP_TARGETS = {
    "__quantum__qis__mz__body",
    "__quantum__qis__m__body",
    "__quantum__qis__measure__body",
    "__quantum__rt__result_record_output",
    "__quantum__rt__result_update_reference_count",
};

static bool isStripTarget(const llvm::CallInst& call) {
  const auto* callee = call.getCalledFunction();
  if (callee == nullptr) {
    return false;
  }
  const auto name = callee->getName();
  return std::ranges::any_of(
      STRIP_TARGETS, [&](llvm::StringRef target) { return name == target; });
}

std::size_t stripMeasurementsAndRecording(llvm::Module& m) {
  std::size_t erased = 0;
  for (auto& fn : m) {
    for (auto& bb : fn) {
      for (auto& inst : llvm::make_early_inc_range(bb)) {
        auto* call = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (call == nullptr || !isStripTarget(*call)) {
          continue;
        }
        // `m` and `measure` return a `Result*`.
        // Replace uses of `m` and `measure` return values with a null before
        // erasure so they do not dangle.
        if (!call->getType()->isVoidTy() && !call->use_empty()) {
          call->replaceAllUsesWith(
              llvm::Constant::getNullValue(call->getType()));
        }
        call->eraseFromParent();
        ++erased;
      }
    }
  }
  return erased;
}

} // namespace qir
