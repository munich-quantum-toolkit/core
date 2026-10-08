/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "qdmi/Client.hpp"

int main() try {
  const qdmi::Session session{};
  dd::Package package{2};
  const auto state = dd::makeZeroState(2, package);
  const auto success = dd::getVector(state) == dd::CVec{1., 0., 0., 0.};
  package.decRef(state);
  return success ? 0 : 1;
} catch (...) {
  return 1;
}
