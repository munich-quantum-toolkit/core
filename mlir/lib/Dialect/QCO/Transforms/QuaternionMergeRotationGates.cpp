/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <iostream>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGEROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief This pattern attempts to merge consecutive rotation gates by using
 * quaternions
 */
struct MergeRotationGates final
    : impl::MergeRotationGatesBase<MergeRotationGates> {
  using impl::MergeRotationGatesBase<
      MergeRotationGates>::MergeRotationGatesBase;

  void runOnOperation() override {
    // TODO implement pass here
  }
};

} // namespace mlir::qco
