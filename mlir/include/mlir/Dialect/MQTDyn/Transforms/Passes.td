// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTD_PASSES
#define MQTD_PASSES

include "mlir/Pass/PassBase.td"

def ConstantFolding : Pass<"constant-folding", "mlir::ModuleOp"> {
  let summary = "This pass performs constant folding for several operands that might also be added as static arguments";
  let description = [{
    This pass transforms constants that are passed to mqtdyn operations as operands into static arguments.

    It is currently compatible with:
    - mqtdyn.extractQubit
  }];
}

#endif // MQTD_PASSES
