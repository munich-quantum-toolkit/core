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
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"

#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace dd {

TEST(DDHistogram, ProjectsEntangledStateWithoutChangingIt) {
  auto packageOwner = ::mqt::test::value(dd::Package::create(3));
  auto& package = *packageOwner;
  const CVec amplitudes{
      {0., 0.},  {0.1, 0.2}, {0.2, 0.1}, {0.3, 0.},
      {0., 0.4}, {0.5, 0.},  {0., 0.6},  {0.2, 0.},
  };
  // Squared amplitudes sum to one; hidden qubits have different suffixes.
  const auto state =
      ::mqt::test::value(makeStateFromVector(amplitudes, package));
  const auto original = ::mqt::test::value(getVector(state));
  const std::vector<SamplingOutput> outputs{Qubit{0}, true, Qubit{2}, Qubit{0}};
  std::map<std::string, double> probabilities;
  for (size_t i = 0; i < amplitudes.size(); ++i) {
    std::string key{'0', '1', '0', '0'};
    key[0] = key[3] = (i & 1U) != 0 ? '1' : '0';
    key[2] = (i & 4U) != 0 ? '1' : '0';
    probabilities[key] += std::norm(amplitudes[i]);
  }
  constexpr size_t shots = 32768;
  std::mt19937_64 rng(17);
  const auto counts =
      ::mqt::test::value(Package::sampleCounts(state, outputs, shots, rng));
  size_t total = 0;
  for (const auto& [key, count] : counts) {
    ASSERT_TRUE(probabilities.contains(key));
    const auto probability = probabilities.at(key);
    EXPECT_NEAR(static_cast<double>(count), shots * probability,
                (6 * std::sqrt(shots * probability * (1 - probability))) + 1);
    total += count;
  }
  EXPECT_EQ(total, shots);
  EXPECT_EQ(::mqt::test::value(getVector(state)), original);
  EXPECT_TRUE(::mqt::test::value(Package::sampleCounts(state, outputs, 0, rng))
                  .empty());
  EXPECT_EQ(::mqt::test::value(Package::sampleCounts(state, {}, shots, rng)),
            (std::map<std::string, size_t>{{"", shots}}));
  const std::vector<SamplingOutput> invalid{Qubit{3}};
  EXPECT_EQ(::mqt::test::errorCode(
                [&] { return Package::sampleCounts(state, invalid, 1, rng); }),
            std::errc::invalid_argument);
  package.decRef(state);
}

TEST(DDHistogram, SamplesWideCorrelatedAndIndependentBits) {
  auto packageOwner = ::mqt::test::value(dd::Package::create(128));
  auto& package = *packageOwner;
  const auto ghz = ::mqt::test::value(makeGHZState(128, package));
  std::vector<SamplingOutput> outputs;
  for (size_t i = 128; i > 0; --i) {
    outputs.emplace_back(static_cast<Qubit>(i - 1));
  }
  std::mt19937_64 rng(17);
  const auto counts =
      ::mqt::test::value(Package::sampleCounts(ghz, outputs, 1000000, rng));
  ASSERT_EQ(counts.size(), 2);
  EXPECT_NEAR(static_cast<double>(counts.at(std::string(128, '0'))), 500000.,
              3000.);
  EXPECT_EQ(counts.at(std::string(128, '0')) + counts.at(std::string(128, '1')),
            1000000);
  rng.seed(17);
  EXPECT_EQ(
      ::mqt::test::value(Package::sampleCounts(ghz, outputs, 1000000, rng)),
      counts);
  package.decRef(ghz);

  const auto uniform = ::mqt::test::value(makeBasisState(
      24, std::vector<BasisStates>(24, BasisStates::plus), package));
  outputs.resize(24);
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i] = static_cast<Qubit>(i);
  }
  constexpr size_t shots = 32768;
  const auto independent =
      ::mqt::test::value(Package::sampleCounts(uniform, outputs, shots, rng));
  size_t total = 0;
  std::vector<size_t> ones(24);
  std::map<std::string, size_t> prefixes;
  for (const auto& [key, count] : independent) {
    ASSERT_EQ(key.size(), 24);
    total += count;
    prefixes[key.substr(0, 4)] += count;
    for (size_t i = 0; i < key.size(); ++i) {
      ones[i] += key[i] == '1' ? count : 0;
    }
  }
  EXPECT_EQ(total, shots);
  for (const auto count : ones) {
    EXPECT_NEAR(static_cast<double>(count), shots / 2.,
                6 * std::sqrt(shots / 4.));
  }
  ASSERT_EQ(prefixes.size(), 16);
  for (const auto& [prefix, count] : prefixes) {
    EXPECT_NEAR(static_cast<double>(count), shots / 16.,
                6 * std::sqrt(shots * 15. / 256.));
  }
  package.decRef(uniform);
}

TEST(DDGateConstruction, AppliesGlobalPhase) {
  auto packageOwner = ::mqt::test::value(Package::create(1));
  auto& package = *packageOwner;
  auto state = ::mqt::test::value(makeZeroState(1, package));

  const auto phased = applyGlobalPhase(state, std::numbers::pi / 2., package);
  EXPECT_EQ(state, phased);
  const auto vector = ::mqt::test::value(dd::getVector(phased));

  ASSERT_EQ(vector.size(), 2);
  EXPECT_NEAR(vector[0].real(), 0., RealNumber::eps);
  EXPECT_NEAR(vector[0].imag(), 1., RealNumber::eps);
  EXPECT_EQ(vector[1], std::complex<fp>{});

  package.garbageCollect(true);
  state = package.applyOperation(Package::makeIdent(), state);
  EXPECT_EQ(::mqt::test::value(dd::getVector(state)), vector);
  package.decRef(state);
  package.garbageCollect(true);
  const auto [vectors, matrices, reals] = package.computeActiveCounts();
  EXPECT_EQ(vectors, 0);
  EXPECT_EQ(matrices, 0);
  EXPECT_EQ(reals, 0);
}

TEST(DDGateConstruction, ScalarGlobalPhaseSurvivesCollection) {
  auto packageOwner = ::mqt::test::value(Package::create(0));
  auto& package = *packageOwner;
  auto state = vEdge::one();
  applyGlobalPhase(state, 0.3, package);
  package.garbageCollect(true);
  EXPECT_NEAR(std::abs(::mqt::test::value(dd::getVector(state)).front() -
                       std::polar(1., 0.3)),
              0., RealNumber::eps);
  package.decRef(state);
  package.garbageCollect(true);
}

TEST(DDGateConstruction, VectorKroneckerWithTerminal) {
  constexpr std::size_t nq = 1;
  constexpr auto root = vEdge::one();
  auto packageOwner = ::mqt::test::value(Package::create(nq));
  auto& package = *packageOwner;

  const auto zeroState = ::mqt::test::value(makeZeroState(nq, package));
  const auto extendedRoot = package.kronecker(zeroState, root, 0);
  EXPECT_EQ(zeroState, extendedRoot);

  package.decRef(zeroState);
  package.garbageCollect(true);

  const auto [vector, matrix, reals] = package.computeActiveCounts();
  EXPECT_EQ(vector, 0);
  EXPECT_EQ(matrix, 0);
  EXPECT_EQ(reals, 0);
}

} // namespace dd
