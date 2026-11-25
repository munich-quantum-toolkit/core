/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"

#include "mlir/IR/Verifier.h"

namespace mlir::qco {

#define GEN_PASS_DEF_VERIFYQCOPROGRAM
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct VerifyQCOProgram final : impl::VerifyQCOProgramBase<VerifyQCOProgram> {
protected:
  void runOnOperation() override {
    if (failed(verify(getOperation())) ||
        failed(verifyLinearity(getOperation()))) {
      signalPassFailure();
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::qco
