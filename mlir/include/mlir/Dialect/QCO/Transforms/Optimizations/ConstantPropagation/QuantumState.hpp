/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_QUANTUMSTATEORTOP_H
#define MQT_CORE_QUANTUMSTATEORTOP_H

namespace mlir::qco {

class QuantumState {
  int q = 1;

public:
  int inc(const int i) const;
};

} // namespace mlir::qco

#endif // MQT_CORE_QUANTUMSTATEORTOP_H
