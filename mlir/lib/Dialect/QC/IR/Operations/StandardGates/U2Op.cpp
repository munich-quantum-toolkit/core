/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::utils;

void U2Op::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubitIn,
                 const std::variant<double, Value>& phi,
                 const std::variant<double, Value>& lambda) {
  auto phiOperand = variantToValue(odsBuilder, odsState.location, phi);
  auto lambdaOperand = variantToValue(odsBuilder, odsState.location, lambda);
  build(odsBuilder, odsState, qubitIn, phiOperand, lambdaOperand);
}
