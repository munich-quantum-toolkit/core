/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUANTUMSTATEORTOP
#define MQT_CORE_QUANTUMSTATEORTOP
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/QuantumState.hpp"

namespace mlir::qco {

int QuantumState::inc(const int i) const { return i + q; }

} // namespace mlir::qco

#endif // MQT_CORE_QUANTUMSTATEORTOP
