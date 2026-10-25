/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QAOA.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace mqt::bench {

// Evaluate the paper's cost eigenvalues, independently of RZZ decomposition.
static dd::CVec qaoaState(const size_t qubits, llvm::ArrayRef<double> gammas,
                          llvm::ArrayRef<double> betas) {
  const auto size = size_t{1} << qubits;
  dd::CVec state(size, 1. / std::sqrt(static_cast<double>(size)));
  for (size_t layer = 0; layer < gammas.size(); ++layer) {
    for (size_t index = 0; index < size; ++index) {
      size_t cut = 0;
      for (size_t bit = 0; bit < qubits; ++bit) {
        const auto neighbor = (bit + 1) % qubits;
        if (((index >> bit) & 1U) != ((index >> neighbor) & 1U)) {
          ++cut;
        }
      }
      state[index] *= std::polar(1., -gammas[layer] * static_cast<double>(cut));
    }
    const auto diagonal = std::cos(betas[layer]);
    const std::complex<double> offDiagonal{0., -std::sin(betas[layer])};
    for (size_t bit = 0; bit < qubits; ++bit) {
      const auto mask = size_t{1} << bit;
      for (size_t index = 0; index < size; ++index) {
        if ((index & mask) != 0) {
          continue;
        }
        const auto zero = state[index];
        const auto one = state[index | mask];
        state[index] = (diagonal * zero) + (offDiagonal * one);
        state[index | mask] = (offDiagonal * zero) + (diagonal * one);
      }
    }
  }
  return state;
}

TEST(GenerateProgramTest, QAOAMatchesCostAndMixerEvolution) {
  struct Angles {
    std::vector<double> gammas;
    std::vector<double> betas;
  };
  for (const size_t qubits : {3U, 4U, 5U}) {
    for (const auto& angles : {
             Angles{
                 .gammas = {std::numbers::pi / 4.},
                 .betas = {std::numbers::pi / 8.},
             },
             Angles{
                 .gammas = {0.31, 1.17, -7.1},
                 .betas = {0.23, 0.49, -3.5},
             },
         }) {
      SCOPED_TRACE(qubits);
      SCOPED_TRACE(angles.gammas.size());
      auto program = test::generateQCO(llvm::cantFail(
          QAOA::create({.qubits = qubits, .depth = angles.gammas.size()})));
      ASSERT_TRUE(mlir::succeeded(program));
      ASSERT_TRUE(mlir::succeeded(program->bindParameters({
          {"gammas", angles.gammas},
          {"betas", angles.betas},
      })));
      dd::Package package(0);
      auto state = mlir::qco::simulateStatevector(
          mlir::mqt::getEntryPoint(program->module()), package);
      ASSERT_TRUE(mlir::succeeded(state));
      ASSERT_EQ(package.qubits(), qubits);
      auto expected = dd::makeStateFromVector(
          qaoaState(qubits, angles.gammas, angles.betas), package);
      EXPECT_NEAR(package.fidelity(*state, expected), 1., 1e-10);
      package.decRef(expected);
      package.decRef(*state);
      if (angles.gammas.size() == 1) {
        ASSERT_TRUE(mlir::succeeded(program->unrollQuantumLoops()));
        ASSERT_TRUE(mlir::succeeded(program->cleanup()));
        llvm::DenseMap<mlir::Value, size_t> depths;
        size_t depth = 0;
        program->module().walk([&](mlir::qco::RZZOp gate) {
          const auto current = 1 + std::max(depths.lookup(gate.getQubit0In()),
                                            depths.lookup(gate.getQubit1In()));
          depths[gate.getQubit0Out()] = current;
          depths[gate.getQubit1Out()] = current;
          depth = std::max(depth, current);
        });
        EXPECT_EQ(depth, qubits % 2 == 0 ? 2U : 3U);
      }
    }
  }
}

TEST(GenerateProgramTest, SamplesQAOACutQuality) {
  for (const size_t qubits : {3U, 4U, 5U}) {
    const auto benchmark = llvm::cantFail(QAOA::create({.qubits = qubits}));
    auto program = test::generateQCO(benchmark);
    ASSERT_TRUE(mlir::succeeded(program));
    ASSERT_TRUE(mlir::succeeded(program->bindParameters({
        {"gammas", std::vector<double>{std::numbers::pi / 4.}},
        {"betas", std::vector<double>{std::numbers::pi / 8.}},
    })));
    auto counts = mlir::qco::sample(mlir::mqt::getEntryPoint(program->module()),
                                    16'384, 17);
    ASSERT_TRUE(mlir::succeeded(counts));
    // The triangle has a different light cone than the longer cycles.
    const auto cutFraction = qubits == 3 ? 5. / 8. : 3. / 4.;
    const auto evaluation = llvm::cantFail(benchmark.evaluate(*counts));
    EXPECT_NEAR(evaluation.meanCut / static_cast<double>(qubits), cutFraction,
                0.03);
    EXPECT_NEAR(evaluation.approximationRatio,
                evaluation.meanCut / static_cast<double>(qubits - qubits % 2),
                1e-12);
  }
}

TEST(GenerateProgramTest, PreservesQAOAInputsThroughJeff) {
  auto program = test::generateQCO(
      llvm::cantFail(QAOA::create({.qubits = 3, .depth = 2})));
  ASSERT_TRUE(mlir::succeeded(program));
  EXPECT_EQ(program->parameters(),
            (std::vector<std::string>{"gammas", "betas"}));
  auto jeff = std::move(*program).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  auto restored = mlir::JeffProgram::fromBytes(jeff->toBytes());
  ASSERT_TRUE(mlir::succeeded(restored));
  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  auto entry = mlir::mqt::getEntryPoint(qco->module());
  EXPECT_EQ(entry.getNumArguments(), 2);
  for (const auto type : entry.getArgumentTypes()) {
    const auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
    ASSERT_TRUE(tensor);
    EXPECT_EQ(tensor.getShape(), (llvm::ArrayRef<int64_t>{2}));
    EXPECT_TRUE(tensor.getElementType().isF64());
  }
}

TEST(GenerateProgramTest, SerializesWideQAOAWithCompactLoops) {
  for (const size_t layers : {2U, 1'000'000'000U}) {
    SCOPED_TRACE(layers);
    const QAOAOptions options{
        .qubits = 1'024,
        .depth = layers,
    };
    auto program = generate(llvm::cantFail(QAOA::create(options)));
    ASSERT_TRUE(mlir::succeeded(program));
    EXPECT_EQ(program->parameters().size(), 2);
    // Generation must not expand a gate list per vertex or layer.
    EXPECT_LT(test::countOperations(program->module()), 150U);
    test::expectJeffRoundTrip(std::move(*program));
  }
}

} // namespace mqt::bench
