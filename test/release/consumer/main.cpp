/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "qdmi/Client.hpp"

#include <memory>

int main() {
  const qdmi::Session session;
  const auto package = std::make_unique<dd::Package>(2);
  const auto state = dd::makeZeroState(2, *package);
  return state.getVector() == dd::CVec{1., 0., 0., 0.} ? 0 : 1;
}
