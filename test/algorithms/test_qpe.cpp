/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/QPE.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <bitset>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>

class QPE : public testing::TestWithParam<std::pair<qc::fp, qc::Qubit>> {
protected:
  qc::fp lambda{};
  qc::Qubit precision{};
  qc::fp theta{};
  bool exactlyRepresentable{};
  std::uint64_t expectedResult{};
  std::string expectedResultRepresentation;
  std::uint64_t secondExpectedResult{};
  std::string secondExpectedResultRepresentation;

  void TearDown() override {}
  void SetUp() override {
    lambda = GetParam().first;
    precision = GetParam().second;

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";

    theta = lambda / 2;

    std::cout << "Expected theta=" << theta << "\n";
    std::bitset<64> binaryExpansion{};
    auto expansion = theta * 2;
    std::size_t index = 0;
    while (std::abs(expansion) > 1e-8) {
      if (expansion >= 1.) {
        binaryExpansion.set(index);
        expansion -= 1.0;
      }
      index++;
      expansion *= 2;
    }

    exactlyRepresentable = true;
    for (std::uint64_t i = precision; i < binaryExpansion.size(); ++i) {
      if (binaryExpansion.test(i)) {
        exactlyRepresentable = false;
        break;
      }
    }

    expectedResult = 0U;
    for (std::uint64_t i = 0; i < precision; ++i) {
      if (binaryExpansion.test(i)) {
        expectedResult |= (1ULL << (precision - 1 - i));
      }
    }
    std::stringstream ss{};
    for (auto i = static_cast<std::int64_t>(precision - 1); i >= 0; --i) {
      if ((expectedResult & (1ULL << i)) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    expectedResultRepresentation = ss.str();

    if (exactlyRepresentable) {
      std::cout << "Theta is exactly representable using " << precision
                << " bits.\n";
      std::cout << "The expected output state is |"
                << expectedResultRepresentation << ">.\n";
    } else {
      secondExpectedResult = expectedResult + 1;
      ss.str("");
      for (auto i = static_cast<std::int64_t>(precision - 1); i >= 0; --i) {
        if ((secondExpectedResult & (1ULL << i)) != 0) {
          ss << 1;
        } else {
          ss << 0;
        }
      }
      secondExpectedResultRepresentation = ss.str();

      std::cout << "Theta is not exactly representable using " << precision
                << " bits.\n";
      std::cout << "Most probable output states are |"
                << expectedResultRepresentation << "> and |"
                << secondExpectedResultRepresentation << ">.\n";
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    QPE, QPE,
    testing::Values(std::pair{1., 1U}, std::pair{0.5, 2U}, std::pair{0.25, 3U},
                    std::pair{3. / 8, 3U}, std::pair{3. / 8, 4U},
                    std::pair{3. / 32, 5U}, std::pair{3. / 32, 6U}),
    [](const testing::TestParamInfo<QPE::ParamType>& inf) {
      // Generate names for test cases
      const auto lambda = inf.param.first;
      const auto precision = inf.param.second;
      std::stringstream ss{};
      ss << static_cast<std::size_t>(lambda * 100) << "_pi_" << precision;
      return ss.str();
    });

TEST_P(QPE, QPETest) {
  auto dd = std::make_unique<dd::Package>(precision + 1);
  auto qc = qc::createQPE(lambda, precision);
  qc.printStatistics(std::cout);
  ASSERT_EQ(qc.getNqubits(), precision + 1);
  ASSERT_NO_THROW({ qc::CircuitOptimizer::removeFinalMeasurements(qc); });

  dd::VectorDD e{};
  ASSERT_NO_THROW(
      { e = dd::simulate(qc, dd->makeZeroState(qc.getNqubits()), *dd); });

  // account for the eigenstate qubit by adding an offset
  const auto offset = 1ULL << (e.p->v + 1);
  const auto amplitude = e.getValueByIndex(expectedResult + offset);
  const auto probability = std::norm(amplitude);
  std::cout << "Obtained probability for |" << expectedResultRepresentation
            << ">: " << probability << "\n";

  if (exactlyRepresentable) {
    EXPECT_NEAR(probability, 1.0, 1e-8);
  } else {
    constexpr auto threshold = 4. / (qc::PI * qc::PI);
    // account for the eigenstate qubit in the expected result by shifting and
    // adding 1
    const auto secondAmplitude =
        e.getValueByIndex(secondExpectedResult + offset);
    const auto secondProbability = std::norm(secondAmplitude);
    std::cout << "Obtained probability for |"
              << secondExpectedResultRepresentation
              << ">: " << secondProbability << "\n";

    EXPECT_GT(probability, threshold);
    EXPECT_GT(secondProbability, threshold);
  }
}

TEST_P(QPE, IQPETest) {
  auto dd = std::make_unique<dd::Package>(precision + 1);
  auto qc = qc::createIterativeQPE(lambda, precision);
  ASSERT_EQ(qc.getNqubits(), 2U);

  constexpr auto shots = 8192U;
  const auto measurements = dd::sample(qc, shots);

  // sort the measurements
  using Measurement = std::pair<std::string, std::size_t>;
  auto comp = [](const Measurement& a, const Measurement& b) -> bool {
    if (a.second != b.second) {
      return a.second > b.second;
    }
    return a.first > b.first;
  };
  const std::set<Measurement, decltype(comp)> ordered(measurements.begin(),
                                                      measurements.end(), comp);

  std::cout << "Obtained measurements: \n";
  for (const auto& [bitstring, count] : ordered) {
    std::cout << "\t" << bitstring << ": " << count << " ("
              << (count * 100) / shots << "%)\n";
  }

  const auto& [mostLikelyResult, mostLikelyCount] = *ordered.begin();
  if (exactlyRepresentable) {
    EXPECT_EQ(mostLikelyResult, expectedResultRepresentation);
    EXPECT_EQ(mostLikelyCount, shots);
  } else {
    auto it = ordered.begin();
    std::advance(it, 1);
    const auto& [secondMostLikelyResult, secondMostLikelyCount] = *(it);
    EXPECT_TRUE(
        (mostLikelyResult == expectedResultRepresentation &&
         secondMostLikelyResult == secondExpectedResultRepresentation) ||
        (mostLikelyResult == secondExpectedResultRepresentation &&
         secondMostLikelyResult == expectedResultRepresentation));
    const auto threshold = 4. / (qc::PI * qc::PI);
    EXPECT_NEAR(static_cast<double>(mostLikelyCount) / shots, threshold, 0.02);
    EXPECT_NEAR(static_cast<double>(secondMostLikelyCount) / shots, threshold,
                0.02);
  }
}

TEST_P(QPE, DynamicEquivalenceSimulation) {
  auto dd = std::make_unique<dd::Package>(precision + 1);

  // create standard QPE circuit
  auto qpe = qc::createQPE(lambda, precision);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(qpe);

  // simulate circuit
  auto e = dd::simulate(qpe, dd->makeZeroState(qpe.getNqubits()), *dd);

  // create standard IQPE circuit
  auto iqpe = qc::createIterativeQPE(lambda, precision);

  // transform dynamic circuits by first eliminating reset operations and
  // afterwards deferring measurements
  qc::CircuitOptimizer::eliminateResets(iqpe);
  qc::CircuitOptimizer::deferMeasurements(iqpe);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(iqpe);

  // simulate circuit
  auto f = dd::simulate(iqpe, dd->makeZeroState(iqpe.getNqubits()), *dd);

  // calculate fidelity between both results
  auto fidelity = dd->fidelity(e, f);
  std::cout << "Fidelity of both circuits: " << fidelity << "\n";

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

TEST_P(QPE, DynamicEquivalenceFunctionality) {
  auto dd = std::make_unique<dd::Package>(precision + 1);

  // create standard QPE circuit
  auto qpe = qc::createQPE(lambda, precision);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(qpe);

  // simulate circuit
  auto e = dd::buildFunctionality(qpe, *dd);

  // create standard IQPE circuit
  auto iqpe = qc::createIterativeQPE(lambda, precision);

  // transform dynamic circuits by first eliminating reset operations and
  // afterwards deferring measurements
  qc::CircuitOptimizer::eliminateResets(iqpe);
  qc::CircuitOptimizer::deferMeasurements(iqpe);
  qc::CircuitOptimizer::backpropagateOutputPermutation(iqpe);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(iqpe);

  // simulate circuit
  auto f = dd::buildFunctionality(iqpe, *dd);

  EXPECT_EQ(e, f);
}
