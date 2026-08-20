/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/QPE.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

class DynamicCircuitEvalExactQPE : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit precision{};
  qc::QuantumComputation qpe;
  qc::QuantumComputation iqpe;
  std::size_t qpeNgates{};
  std::size_t iqpeNgates{};
  std::unique_ptr<dd::Package> dd;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package>(precision + 1);

    const auto lambda = std::ldexp(1., 1 - static_cast<int>(precision));
    qpe = qc::createQPE(lambda, precision);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(qpe);
    qpeNgates = qpe.getNindividualOps();

    iqpe = qc::createIterativeQPE(lambda, precision);
    iqpeNgates = iqpe.getNindividualOps();

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";
  }
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalExactQPE, testing::Range<qc::Qubit>(1U, 64U, 5U),
    [](const testing::TestParamInfo<DynamicCircuitEvalExactQPE::ParamType>&
           inf) {
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

TEST_P(DynamicCircuitEvalExactQPE, UnitaryTransformation) {
  qpe.reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(iqpe);
  qc::CircuitOptimizer::deferMeasurements(iqpe);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(iqpe);
  iqpe.reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  dd::MatrixDD e = dd::Package::makeIdent();
  dd->incRef(e);

  auto leftIt = qpe.begin();
  auto rightIt = iqpe.begin();

  while (leftIt != qpe.end() && rightIt != iqpe.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    auto multRight = dd->multiply(multLeft, getInverseDD(**rightIt, *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != qpe.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != iqpe.end()) {
    auto multRight = dd->multiply(e, getInverseDD(**rightIt, *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++rightIt;
  }
  const auto finishedEC = std::chrono::steady_clock::now();

  const auto preprocessing =
      std::chrono::duration<double>(finishedTransformation - start).count();
  const auto verification =
      std::chrono::duration<double>(finishedEC - finishedTransformation)
          .count();

  std::stringstream ss{};
  ss << "qpe_exact,transformation," << qpe.getNqubits() << "," << qpeNgates
     << ",2," << iqpeNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

namespace {

class DynamicCircuitEvalInexactQPE : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit precision{};
  qc::QuantumComputation qpe;
  qc::QuantumComputation iqpe;
  std::size_t qpeNgates{};
  std::size_t iqpeNgates{};
  std::unique_ptr<dd::Package> dd;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package>(precision + 1);

    const auto lambda = std::ldexp(3., -static_cast<int>(precision));
    qpe = qc::createQPE(lambda, precision);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(qpe);
    qpeNgates = qpe.getNindividualOps();

    iqpe = qc::createIterativeQPE(lambda, precision);
    iqpeNgates = iqpe.getNindividualOps();

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";
  }
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalInexactQPE, testing::Range<qc::Qubit>(1U, 15U, 3U),
    [](const testing::TestParamInfo<DynamicCircuitEvalInexactQPE::ParamType>&
           inf) {
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

TEST_P(DynamicCircuitEvalInexactQPE, UnitaryTransformation) {
  qpe.reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(iqpe);
  qc::CircuitOptimizer::deferMeasurements(iqpe);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(iqpe);
  iqpe.reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  dd::MatrixDD e = dd::Package::makeIdent();
  dd->incRef(e);

  auto leftIt = qpe.begin();
  auto rightIt = iqpe.begin();

  while (leftIt != qpe.end() && rightIt != iqpe.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    auto multRight = dd->multiply(multLeft, getInverseDD(**rightIt, *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != qpe.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != iqpe.end()) {
    auto multRight = dd->multiply(e, getInverseDD(**rightIt, *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++rightIt;
  }
  const auto finishedEC = std::chrono::steady_clock::now();

  const auto preprocessing =
      std::chrono::duration<double>(finishedTransformation - start).count();
  const auto verification =
      std::chrono::duration<double>(finishedEC - finishedTransformation)
          .count();

  std::stringstream ss{};
  ss << "qpe_inexact,transformation," << qpe.getNqubits() << "," << qpeNgates
     << ",2," << iqpeNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

namespace {

[[nodiscard]] auto makePalindromicBitString(const qc::Qubit width)
    -> std::string {
  std::string bitString(width, '0');
  for (std::size_t i = 0; i < (width + 1U) / 2U; ++i) {
    if (i % 3U == 1U) {
      continue;
    }
    bitString[i] = '1';
    bitString[width - 1U - i] = '1';
  }
  return bitString;
}

class DynamicCircuitEvalBV : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit bitwidth{};
  qc::QuantumComputation bv;
  qc::QuantumComputation dbv;
  std::string expected;

  void TearDown() override {}
  void SetUp() override {
    bitwidth = GetParam();
    expected = makePalindromicBitString(bitwidth);
    const auto hiddenString = qc::BVBitString(expected);
    bv = qc::createBernsteinVazirani(hiddenString, bitwidth);
    dbv = qc::createIterativeBernsteinVazirani(hiddenString, bitwidth);
    std::cout << "Hidden bitstring: " << expected << " (" << bitwidth
              << " qubits)\n";
  }
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalBV, testing::Range<qc::Qubit>(1U, 64U, 5U),
    [](const testing::TestParamInfo<DynamicCircuitEvalBV::ParamType>& inf) {
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

TEST_P(DynamicCircuitEvalBV, ObservableResult) {
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(dbv);
  qc::CircuitOptimizer::deferMeasurements(dbv);

  constexpr std::size_t shots = 128U;
  constexpr std::size_t seed = 7U;
  const auto ordinaryMeasurements = dd::sample(bv, shots, seed);
  const auto iterativeMeasurements = dd::sample(dbv, shots, seed);

  EXPECT_EQ(ordinaryMeasurements, iterativeMeasurements);
}

namespace {

class DynamicCircuitEvalQFT : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit precision{};
  qc::QuantumComputation qft;
  qc::QuantumComputation dqft;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();
    qft = qc::createQFT(precision);
    dqft = qc::createIterativeQFT(precision);
  }
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalQFT, testing::Range<qc::Qubit>(1U, 65U, 5U),
    [](const testing::TestParamInfo<DynamicCircuitEvalQFT::ParamType>& inf) {
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

TEST_P(DynamicCircuitEvalQFT, ObservableDistribution) {
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(dqft);
  qc::CircuitOptimizer::deferMeasurements(dqft);

  constexpr std::size_t shots = 256U;
  constexpr std::size_t seed = 7U;
  const auto ordinaryMeasurements = dd::sample(qft, shots, seed);
  const auto iterativeMeasurements = dd::sample(dqft, shots, seed);
  const auto maxUnique = std::min<std::size_t>(1ULL << precision, shots);

  const auto ordinaryRatio = static_cast<double>(ordinaryMeasurements.size()) /
                             static_cast<double>(maxUnique);
  const auto iterativeRatio =
      static_cast<double>(iterativeMeasurements.size()) /
      static_cast<double>(maxUnique);

  EXPECT_GE(ordinaryRatio, 0.7);
  EXPECT_GE(iterativeRatio, 0.7);
}
