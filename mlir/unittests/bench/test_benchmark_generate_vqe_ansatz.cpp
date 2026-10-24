/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"
#include "bench/VQEAnsatz.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace mqt::bench {

// Apply each full rotation matrix and the chain's prefix-parity permutation.
static dd::CVec vqeAnsatzState(const size_t qubits,
                               llvm::ArrayRef<double> thetas) {
  const auto size = size_t{1} << qubits;
  dd::CVec state(size);
  state[0] = 1.;
  const auto rotationLayers = thetas.size() / qubits;
  for (size_t layer = 0; layer < rotationLayers; ++layer) {
    dd::CVec rotated(size);
    for (size_t output = 0; output < size; ++output) {
      for (size_t input = 0; input < size; ++input) {
        auto element = 1.;
        for (size_t bit = 0; bit < qubits; ++bit) {
          const auto angle = thetas[(layer * qubits) + bit] / 2.;
          const auto inputBit = (input >> bit) & 1U;
          const auto outputBit = (output >> bit) & 1U;
          element *= inputBit == outputBit
                         ? std::cos(angle)
                         : (inputBit == 0 ? std::sin(angle) : -std::sin(angle));
        }
        rotated[output] += element * state[input];
      }
    }
    if (layer + 1 == rotationLayers) {
      return rotated;
    }
    for (size_t input = 0; input < size; ++input) {
      size_t output = 0;
      size_t parity = 0;
      for (size_t bit = 0; bit < qubits; ++bit) {
        parity ^= (input >> bit) & 1U;
        output |= parity << bit;
      }
      state[output] = rotated[input];
    }
  }
  return state;
}

TEST(GenerateProgramTest, VQEAnsatzMatchesRealAmplitudesEvolution) {
  for (const size_t qubits : {1U, 2U, 3U, 4U}) {
    for (const size_t depth : {1U, 3U}) {
      SCOPED_TRACE(qubits);
      SCOPED_TRACE(depth);
      const VQEAnsatz benchmark{{.qubits = qubits, .depth = depth}};
      std::vector<double> thetas(benchmark.numParameters());
      for (size_t index = 0; index < thetas.size(); ++index) {
        thetas[index] =
            static_cast<double>(index + 1) * (index % 2 == 0 ? 0.17 : -0.23);
      }
      auto program = test::generateQCO(benchmark);
      ASSERT_TRUE(mlir::succeeded(program));
      ASSERT_TRUE(
          mlir::succeeded(program->bindParameters({{"thetas", thetas}})));
      dd::Package package(0);
      auto state = mlir::qco::simulateStatevector(
          mlir::mqt::getEntryPoint(program->module()), package);
      ASSERT_TRUE(mlir::succeeded(state));
      ASSERT_EQ(package.qubits(), qubits);
      auto expected =
          dd::makeStateFromVector(vqeAnsatzState(qubits, thetas), package);
      EXPECT_NEAR(package.fidelity(*state, expected), 1., 1e-10);
      package.decRef(expected);
      package.decRef(*state);
    }
  }
}

TEST(GenerateProgramTest, SamplesVQEAnsatzRotationsAndEntanglement) {
  struct Case {
    VQEAnsatzOptions options;
    std::vector<double> thetas;
    std::string outcome;
  };
  for (const auto& example : {
           Case{
               .options = {.qubits = 1, .depth = 2},
               .thetas =
                   {
                       std::numbers::pi / 4.,
                       std::numbers::pi / 4.,
                       std::numbers::pi / 2.,
                   },
               .outcome = "1",
           },
           Case{
               .options = {.qubits = 3},
               .thetas = {std::numbers::pi, 0., 0., 0., 0., 0.},
               .outcome = "111",
           },
           Case{
               .options = {.qubits = 3},
               .thetas = {0., 0., 0., 0., std::numbers::pi, 0.},
               .outcome = "010",
           },
       }) {
    SCOPED_TRACE(example.outcome);
    auto program = test::generateQCO(VQEAnsatz{example.options});
    ASSERT_TRUE(mlir::succeeded(program));
    ASSERT_TRUE(
        mlir::succeeded(program->bindParameters({{"thetas", example.thetas}})));
    auto counts =
        mlir::qco::sample(mlir::mqt::getEntryPoint(program->module()), 128, 17);
    ASSERT_TRUE(mlir::succeeded(counts));
    EXPECT_EQ(*counts, (Counts{{example.outcome, 128}}));
  }
}

TEST(GenerateProgramTest, SamplesBoundVQEAnsatzAfterJeffRoundTrip) {
  auto program = test::generateQCO(VQEAnsatz{{.qubits = 3, .depth = 2}});
  ASSERT_TRUE(mlir::succeeded(program));
  EXPECT_EQ(program->parameters(), (std::vector<std::string>{"thetas"}));
  std::vector<double> thetas(9);
  thetas[6] = std::numbers::pi;
  ASSERT_TRUE(mlir::succeeded(program->bindParameters({{"thetas", thetas}})));
  auto jeff = std::move(*program).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  auto restored = mlir::JeffProgram::fromBytes(jeff->toBytes());
  ASSERT_TRUE(mlir::succeeded(restored));
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  auto counts =
      mlir::qco::sample(mlir::mqt::getEntryPoint(qco->module()), 128, 17);
  ASSERT_TRUE(mlir::succeeded(counts));
  EXPECT_EQ(*counts, (Counts{{"001", 128}}));
}

TEST(GenerateProgramTest, GeneratesWideVQEAnsatzWithCompactLoops) {
  for (const size_t depth : {2U, 1'000'000'000U}) {
    SCOPED_TRACE(depth);
    auto program = generate(VQEAnsatz{{.qubits = 1'024, .depth = depth}});
    ASSERT_TRUE(mlir::succeeded(program));
    EXPECT_EQ(program->parameters().size(), 1U);
    // Generation must not expand a gate list per qubit or layer.
    EXPECT_LT(test::countOperations(program->module()), 150U);
  }
}

TEST(GenerateProgramTest, SerializesWideVQEAnsatzWithCompactLoops) {
  // jeff uses signed 32-bit indices; keep the flattened angle index in range.
  auto program = generate(VQEAnsatz{{.qubits = 1'024, .depth = 1'000'000}});
  ASSERT_TRUE(mlir::succeeded(program));
  test::expectJeffRoundTrip(std::move(*program));
}

} // namespace mqt::bench
