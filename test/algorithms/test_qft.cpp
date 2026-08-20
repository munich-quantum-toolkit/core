/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/QFT.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>

namespace {

class QFT : public testing::TestWithParam<qc::Qubit> {
protected:
  void TearDown() override {}

  void SetUp() override {
    nqubits = GetParam();
    dd = std::make_unique<dd::Package>(nqubits);
  }

  qc::Qubit nqubits = 0;
  std::unique_ptr<dd::Package> dd;
  qc::QuantumComputation qc;
  dd::VectorDD sim{};
  dd::MatrixDD func{};
};

} // namespace

/// Findings from the QFT Benchmarks:
/// The DDpackage has to be able to represent all 2^n different amplitudes in
/// order to produce correct results The smallest entry seems to be closely
/// related to '1-cos(pi/2^(n-1))' The following CN::TOLERANCE values suffice up
/// until a certain number of qubits: 	10e-10	..	18 qubits
///		10e-11	..	20 qubits
///		10e-12	..	22 qubits
///		10e-13	..	23 qubits
/// The accuracy of double floating points allows for a minimal CN::TOLERANCE
/// value of 10e-15
constexpr qc::Qubit QFT_MAX_QUBITS = 17U;

constexpr size_t INITIAL_COMPLEX_COUNT = dd::immortals::size();

[[nodiscard]] auto mapBasisIndex(const uint64_t index, const qc::Qubit nqubits,
                                 const qc::Permutation& permutation)
    -> uint64_t {
  auto mapped = uint64_t{0U};
  for (auto logical = qc::Qubit{0U}; logical < nqubits; ++logical) {
    const auto position = permutation.find(logical);
    const auto physical =
        position == permutation.end() ? logical : position->second;
    if ((index & (uint64_t{1U} << logical)) != 0U) {
      mapped |= uint64_t{1U} << physical;
    }
  }
  return mapped;
}

void expectQFTMatrix(const qc::QuantumComputation& computation,
                     const dd::MatrixDD& matrix, const dd::Package& package) {
  const auto dimension = uint64_t{1U} << computation.getNqubits();
  const auto amplitude = 1.0 / std::sqrt(static_cast<double>(dimension));

  for (auto output = uint64_t{0U}; output < dimension; ++output) {
    for (auto logicalInput = uint64_t{0U}; logicalInput < dimension;
         ++logicalInput) {
      SCOPED_TRACE(testing::Message() << "output " << output
                                      << ", logical input " << logicalInput);
      const auto storedInput =
          mapBasisIndex(logicalInput, computation.getNqubits(),
                        computation.outputPermutation);
      const auto phase = 2.0 * std::numbers::pi *
                         static_cast<double>(output * logicalInput) /
                         static_cast<double>(dimension);
      const auto value =
          matrix.getValueByIndex(package.qubits(), output, storedInput);
      EXPECT_NEAR(value.real(), amplitude * std::cos(phase),
                  dd::RealNumber::eps);
      EXPECT_NEAR(value.imag(), amplitude * std::sin(phase),
                  dd::RealNumber::eps);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(QFT, QFT,
                         testing::Range<qc::Qubit>(0U, QFT_MAX_QUBITS + 1U, 3U),
                         [](const testing::TestParamInfo<QFT::ParamType>& inf) {
                           const auto nqubits = inf.param;
                           std::stringstream ss{};
                           ss << nqubits;
                           if (nqubits == 1) {
                             ss << "_qubit";
                           } else {
                             ss << "_qubits";
                           }
                           return ss.str();
                         });

TEST(QFTSemantic, FullMatrix) {
  constexpr qc::Qubit maxQubits = 4U;
  for (auto nqubits = qc::Qubit{1U}; nqubits <= maxQubits; ++nqubits) {
    const auto computation = qc::createQFT(nqubits, false);
    auto package = dd::Package(nqubits);
    const auto matrix = buildFunctionality(computation, package);

    expectQFTMatrix(computation, matrix, package);

    package.decRef(matrix);
  }
}

TEST_P(QFT, Functionality) {
  qc = qc::createQFT(nqubits, false);
  func = buildFunctionality(qc, *dd);

  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, FunctionalityRecursive) {
  qc = qc::createQFT(nqubits, false);
  func = buildFunctionalityRecursive(qc, *dd);

  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, Simulation) {
  qc = qc::createQFT(nqubits, false);
  const auto in = makeZeroState(nqubits, *dd);
  sim = simulate(qc, in, *dd);

  // first column should consist only of (1/sqrt(2))^n entries
  for (std::uint64_t i = 0; i < 1ULL << nqubits; ++i) {
    auto c = sim.getValueByIndex(i);
    EXPECT_NEAR(
        c.real(),
        static_cast<dd::fp>(std::pow(1.0 / std::numbers::sqrt2, nqubits)),
        dd::RealNumber::eps);
  }
  dd->decRef(sim);
  dd->garbageCollect(true);
}

TEST_P(QFT, FunctionalityRecursiveEquality) {
  qc = qc::createQFT(nqubits, false);
  func = buildFunctionalityRecursive(qc, *dd);
  dd::MatrixDD funcRec{};
  funcRec = buildFunctionality(qc, *dd);

  ASSERT_EQ(func, funcRec);
  dd->decRef(funcRec);
  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, SimulationSampling) {
  const auto dynamic = {false, true};
  for (const auto dyn : dynamic) {
    if (dyn) {
      qc = qc::createIterativeQFT(nqubits);
    } else {
      qc = qc::createQFT(nqubits, false);
    }

    // simulate the circuit
    constexpr std::size_t shots = 8192U;
    const auto measurements = dd::sample(qc, shots);

    const std::size_t unique = measurements.size();
    const auto maxUnique = std::min<std::size_t>(1ULL << nqubits, shots);
    const auto ratio =
        static_cast<double>(unique) / static_cast<double>(maxUnique);

    std::cout << "Unique entries " << unique << " out of " << maxUnique
              << " for a ratio of: " << ratio << "\n";

    // the number of unique entries should be close to the number of shots
    EXPECT_GE(ratio, 0.7);
  }
}
