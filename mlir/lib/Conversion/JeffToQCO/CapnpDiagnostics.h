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

#include "kj/exception.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

namespace mlir::mqt {

/// Report recoverable Cap'n Proto errors without throwing exceptions.
///
/// Keep the handler alive through message destruction and discard output on
/// failure. Fatal Cap'n Proto errors terminate.
class CapnpDiagnosticHandler final : public kj::ExceptionCallback {
public:
  CapnpDiagnosticHandler(Location location, StringRef action)
      : location_(location), action_(action) {}

  void onRecoverableException(kj::Exception&& error) override {
    if (succeeded(result_)) {
      auto exception = std::move(error);
      result_ = failure();
      emitError(location_) << "failed to " << action_ << ": "
                           << exception.getDescription().cStr();
    }
  }

  void onFatalException(kj::Exception&& error) override {
    auto exception = std::move(error);
    llvm::report_fatal_error(exception.getDescription().cStr());
  }

  [[nodiscard]] LogicalResult result() const { return result_; }

private:
  Location location_;
  StringRef action_;
  LogicalResult result_ = success();
};

} // namespace mlir::mqt
