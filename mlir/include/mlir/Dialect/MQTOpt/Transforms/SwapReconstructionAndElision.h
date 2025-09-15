/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>

namespace mqt::ir::opt {

#define GEN_PASS_DECL_SWAPRECONSTRUCTIONANDELISION
#include "SwapReconstructionAndElision.h.inc"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"
} // namespace mqt::ir::opt
