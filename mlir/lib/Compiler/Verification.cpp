/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Verification.h"

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"

#include <stop_token>

namespace mlir {
static FailureOr<qco::DDArgumentBindings>
bindArguments(func::FuncOp entry, const EntryArgumentBindings& bindings) {
  qco::DDArgumentBindings result;
  for (const auto& [index, attribute] : bindings) {
    if (index >= entry.getNumArguments()) {
      return entry.emitError() << "verification binding index " << index
                               << " is outside the entry argument list";
    }
    if (!attribute || attribute.getContext() != entry.getContext()) {
      return entry.emitError() << "verification binding attributes must belong "
                                  "to the entry context";
    }
    result.try_emplace(entry.getArgument(index), attribute);
  }
  return result;
}

FailureOr<qco::VerificationResult>
verify(const QCOProgram& left, const QCOProgram& right,
       const qco::VerificationContract& contract,
       const EntryArgumentBindings& leftBindings,
       const EntryArgumentBindings& rightBindings,
       const qco::VerificationOptions& options, const std::stop_token& stop) {
  if (!left.isValid() || !right.isValid()) {
    const auto context = left.context() ? left.context() : right.context();
    if (context) {
      emitError(UnknownLoc::get(context.get()),
                "verification requires two unconsumed QCO programs");
    }
    return failure();
  }
  auto leftEntry = mqt::getEntryPoint(left.module());
  auto rightEntry = mqt::getEntryPoint(right.module());
  if (!leftEntry || !rightEntry) {
    return emitError(
        UnknownLoc::get(left.context().get()),
        "verification requires a marked entry function in each program");
  }
  auto leftArguments = bindArguments(leftEntry, leftBindings);
  if (failed(leftArguments)) {
    return failure();
  }
  auto rightArguments = bindArguments(rightEntry, rightBindings);
  if (failed(rightArguments)) {
    return failure();
  }
  return qco::verify(leftEntry, rightEntry, contract, *leftArguments,
                     *rightArguments, options, stop);
}
} // namespace mlir
