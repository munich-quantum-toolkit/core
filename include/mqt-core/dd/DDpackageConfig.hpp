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

#include <cstddef>

namespace dd {
struct DDPackageConfig {
  std::size_t utVecNumBucket = 32768U;
  std::size_t utVecInitialAllocationSize = 2048U;
  std::size_t utMatNumBucket = 32768U;
  std::size_t utMatInitialAllocationSize = 2048U;
  std::size_t ctVecAddNumBucket = 16384U;
  std::size_t ctMatAddNumBucket = 16384U;
  std::size_t ctVecAddMagNumBucket = 16384U;
  std::size_t ctMatAddMagNumBucket = 16384U;
  std::size_t ctVecConjNumBucket = 4096U;
  std::size_t ctMatConjTransNumBucket = 4096U;
  std::size_t ctMatVecMultNumBucket = 16384U;
  std::size_t ctMatMatMultNumBucket = 16384U;
  std::size_t ctVecKronNumBucket = 4096U;
  std::size_t ctMatKronNumBucket = 4096U;
  std::size_t ctMatTraceNumBucket = 4096U;
  std::size_t ctVecInnerProdNumBucket = 4096U;
};

constexpr auto UNITARY_SIMULATOR_DD_PACKAGE_CONFIG = []() {
  DDPackageConfig config{};
  config.utMatNumBucket = 65'536U;
  config.ctMatAddNumBucket = 65'536U;
  config.ctMatMatMultNumBucket = 65'536U;
  config.utVecNumBucket = 1U;
  config.utVecInitialAllocationSize = 1U;
  config.ctVecAddNumBucket = 1U;
  config.ctVecConjNumBucket = 1U;
  config.ctMatConjTransNumBucket = 1U;
  config.ctMatVecMultNumBucket = 1U;
  config.ctVecKronNumBucket = 1U;
  config.ctMatKronNumBucket = 1U;
  config.ctMatTraceNumBucket = 1U;
  config.ctVecInnerProdNumBucket = 1U;
  return config;
}();
} // namespace dd
