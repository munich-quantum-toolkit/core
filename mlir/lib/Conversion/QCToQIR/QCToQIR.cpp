/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQIR/QCToQIR.h"

#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"

#include <mlir/Pass/PassManager.h>

void mlir::populateQIRConversionPipeline(mlir::PassManager& pm,
                                         bool useAdaptive) {
  if (useAdaptive) {
    pm.addPass(createQCToQIRAdaptive());
  } else {
    pm.addPass(createQCToQIRBase());
  }
  pm.addPass(qir::createAttachQIRAttributes(
      qir::AttachQIRAttributesOptions{useAdaptive}));
}
