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

#include <llvm/Config/llvm-config.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace mlir::qco::detail {

template <typename Target, typename Inputs>
RegionSuccessor makeRegionSuccessor(Target* target,
                                    [[maybe_unused]] Inputs inputs) {
#if LLVM_VERSION_MAJOR >= 23
  return RegionSuccessor(target);
#else
  return RegionSuccessor(target, inputs);
#endif
}

[[nodiscard]] inline bool isOperationSuccessor(RegionSuccessor successor) {
#if LLVM_VERSION_MAJOR >= 23
  return successor.isOperation();
#else
  return successor.isParent();
#endif
}

} // namespace mlir::qco::detail
