/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/operations/OpType.hpp"

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
  std::size_t ctDmTraceNumBucket = 1U;
  std::size_t ctMatTraceNumBucket = 4096U;
  std::size_t ctVecInnerProdNumBucket = 4096U;
  std::size_t ctDmNoiseNumBucket = 1U;
  std::size_t utDmNumBucket = 1U;
  std::size_t utDmInitialAllocationSize = 1U;
  std::size_t ctDmDmMultNumBucket = 1U;
  std::size_t ctDmAddNumBucket = 1U;

  // The number of different quantum operations. i.e., the number of operations
  // defined in OpType.hpp. This parameter is required to initialize the
  // StochasticNoiseOperationTable.hpp
  std::size_t stochasticCacheOps = 1;
};

constexpr auto STOCHASTIC_NOISE_SIMULATOR_DD_PACKAGE_CONFIG = []() {
  DDPackageConfig config{};
  config.stochasticCacheOps = qc::OpType::OpTypeEnd;
  config.ctVecAddMagNumBucket = 1U;
  config.ctMatAddMagNumBucket = 1U;
  config.ctVecConjNumBucket = 1U;
  return config;
}();

constexpr auto DENSITY_MATRIX_SIMULATOR_DD_PACKAGE_CONFIG = []() {
  DDPackageConfig config{};
  config.utDmNumBucket = 65536U;
  config.utDmInitialAllocationSize = 4096U;
  config.ctDmDmMultNumBucket = 16384U;
  config.ctDmAddNumBucket = 16384U;
  config.ctDmNoiseNumBucket = 4096U;
  config.utMatNumBucket = 16384U;
  config.ctMatAddNumBucket = 4096U;
  config.ctVecAddNumBucket = 4096U;
  config.ctMatConjTransNumBucket = 4096U;
  config.ctMatMatMultNumBucket = 1U;
  config.ctMatVecMultNumBucket = 1U;
  config.utVecNumBucket = 1U;
  config.utVecInitialAllocationSize = 1U;
  config.utMatInitialAllocationSize = 1U;
  config.ctVecKronNumBucket = 1U;
  config.ctMatKronNumBucket = 1U;
  config.ctDmTraceNumBucket = 4096U;
  config.ctMatTraceNumBucket = 1U;
  config.ctVecInnerProdNumBucket = 1U;
  config.stochasticCacheOps = 1U;
  config.ctVecAddMagNumBucket = 1U;
  config.ctMatAddMagNumBucket = 1U;
  config.ctVecConjNumBucket = 1U;
  return config;
}();

constexpr auto UNITARY_SIMULATOR_DD_PACKAGE_CONFIG = []() {
  DDPackageConfig config{};
  config.utMatNumBucket = 65'536U;
  config.ctMatAddNumBucket = 65'536U;
  config.ctMatMatMultNumBucket = 65'536U;
  config.utVecNumBucket = 1U;
  config.utVecInitialAllocationSize = 1U;
  config.ctVecAddNumBucket = 1U;
  config.ctVecAddMagNumBucket = 1U;
  config.ctMatAddMagNumBucket = 1U;
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
