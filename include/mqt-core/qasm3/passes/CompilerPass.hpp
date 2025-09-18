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

namespace qasm3 {
class Statement;

class CompilerPass {
public:
  virtual ~CompilerPass() = default;

  virtual void processStatement(Statement& statement) = 0;
};
} // namespace qasm3
