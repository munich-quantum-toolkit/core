/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/WState.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/Edge.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <utility>

namespace mqt::bench {

TEST(GenerateProgramTest, WStatePreservesCoherenceAndStructuredLoops) {
  for (const size_t qubits : {1U, 3U, 16U}) {
    const WState benchmark{{.qubits = qubits}};
    auto qc = generate(benchmark);
    ASSERT_TRUE(qc);
    if (qubits > 1) {
      EXPECT_GT(test::countOps<mlir::scf::ForOp>(qc->module()), 0U);
      const auto angles = test::angleTable(qc->module());
      ASSERT_TRUE(angles);
      EXPECT_EQ(angles.getNumElements(), qubits - 1);
    }
    auto qco = qc->copy().intoQCO();
    ASSERT_TRUE(qco);
    const auto before = qco->str();
    dd::Package package(0);
    const auto root = mlir::qco::simulateStatevector(
        mlir::mqt::getEntryPoint(qco->module()), package);
    ASSERT_TRUE(mlir::succeeded(root));
    EXPECT_EQ(qco->str(), before);
    qco.reset();
    EXPECT_NEAR(package.fidelity(*root, dd::makeWState(qubits, package)), 1.,
                1e-10);
    for (size_t wire = 0; wire < qubits; ++wire) {
      auto outcome = std::string(qubits, '0');
      outcome[qubits - wire - 1] = '1';
      const auto amplitude = root->getValueByPath(qubits, outcome);
      EXPECT_NEAR(amplitude.real(), 1. / std::sqrt(static_cast<double>(qubits)),
                  1e-10);
      EXPECT_NEAR(amplitude.imag(), 0., 1e-10);
    }
    test::expectJeffRoundTrip(std::move(*qc));
    test::expectSamplingMatchesReference(benchmark);
  }
}

TEST(GenerateProgramTest, Simulates4096QubitWStateWithoutDenseExtraction) {
  constexpr size_t qubits = 4096;
  const WState benchmark{{.qubits = qubits}};
  auto qc = generate(benchmark);
  ASSERT_TRUE(qc);
  EXPECT_LT(test::countOperations(qc->module()), 100U);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(jeff);
  auto restored = mlir::JeffProgram::fromBytes(jeff->toBytes());
  ASSERT_TRUE(restored);
  qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(qco);
  dd::Package package(0);
  const auto root = mlir::qco::simulateStatevector(
      mlir::mqt::getEntryPoint(qco->module()), package);
  ASSERT_TRUE(mlir::succeeded(root));
  EXPECT_EQ(package.qubits(), qubits);
  EXPECT_NEAR(package.fidelity(*root, dd::makeWState(qubits, package)), 1.,
              1e-8);
  EXPECT_NEAR(package.innerProduct(*root, *root).r, 1., 1e-8);
  for (const size_t index : {0U, 2047U, 4095U}) {
    auto outcome = std::string(qubits, '0');
    outcome[index] = '1';
    const auto amplitude = root->getValueByPath(qubits, outcome);
    EXPECT_NEAR(amplitude.real(), 1. / 64., 1e-8);
    EXPECT_NEAR(amplitude.imag(), 0., 1e-8);
  }
}

} // namespace mqt::bench
