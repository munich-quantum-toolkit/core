// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MergeRotationGates : Pass<"merge-rotation-gates",
"mlir::ModuleOp"> {
  let summary = "Merge rotation gates using quaternion-based fusion";
  let description = [{
    Merges consecutive rotation gates of different types (rx, ry, rz, u)
    using quaternion representation and the Hamilton product.
  }];

  let dependentDialects = ["::mlir::math::MathDialect"];
}
