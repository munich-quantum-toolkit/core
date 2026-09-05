/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.h"
#include "bench/GHZ.hpp"
#include "mlir/bench/Generate.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace mqt::bench {

TEST(GenerateProgramTest, EmitsConfiguredGHZWithoutEagerRegisterLoads) {
  const GHZ benchmark(
      {.qubits = 64, .topology = GHZTopology::Star, .basis = GHZBasis::X});
  auto program = generate(benchmark);
  ASSERT_TRUE(program);

  EXPECT_LT(test::countOps<mlir::memref::LoadOp>(program->module()), 10U);
}

} // namespace mqt::bench
