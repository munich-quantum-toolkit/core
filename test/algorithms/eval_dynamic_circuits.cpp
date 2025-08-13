/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <bitset>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

class DynamicCircuitEvalExactQPE : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit precision{};
  qc::fp theta{};
  std::size_t expectedResult{};
  std::string expectedResultRepresentation;
  qc::QuantumComputation qpe;
  qc::QuantumComputation iqpe;
  std::size_t qpeNgates{};
  std::size_t iqpeNgates{};
  std::unique_ptr<dd::Package> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package>(precision + 1);

    qpe = qc::createQPE(precision);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(qpe);
    qpeNgates = qpe.getNindividualOps();

    // extract lambda from QPE global phase
    const auto lambda = qpe.getGlobalPhase();

    iqpe = qc::createIterativeQPE(lambda, precision);
    iqpeNgates = iqpe.getNindividualOps();

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";

    theta = lambda / 2;

    std::cout << "Expected theta=" << theta << "\n";
    std::bitset<64> binaryExpansion{};
    dd::fp expansion = theta * 2;
    std::size_t index = 0;
    while (std::abs(expansion) > 1e-8) {
      if (expansion >= 1.) {
        binaryExpansion.set(index);
        expansion -= 1.0;
      }
      index++;
      expansion *= 2;
    }

    expectedResult = 0ULL;
    for (std::size_t i = 0; i < precision; ++i) {
      if (binaryExpansion.test(i)) {
        expectedResult |= (1ULL << (precision - 1 - i));
      }
    }
    std::stringstream ss{};
    for (auto i = static_cast<int>(precision - 1); i >= 0; --i) {
      if ((expectedResult & (1ULL << static_cast<std::size_t>(i))) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    expectedResultRepresentation = ss.str();

    std::cout << "Theta is exactly representable using " << precision
              << " bits.\n";
    std::cout << "The expected output state is |"
              << expectedResultRepresentation << ">.\n";
  }
};

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

TEST_P(DynamicCircuitEvalExactQPE, DISABLED_UnitaryTransformation) {
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

class DynamicCircuitEvalInexactQPE : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit precision{};
  dd::fp theta{};
  std::size_t expectedResult{};
  std::string expectedResultRepresentation;
  std::size_t secondExpectedResult{};
  std::string secondExpectedResultRepresentation;
  qc::QuantumComputation qpe;
  qc::QuantumComputation iqpe;
  std::size_t qpeNgates{};
  std::size_t iqpeNgates{};
  std::unique_ptr<dd::Package> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package>(precision + 1);

    qpe = qc::createQPE(precision, false);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(qpe);
    qpeNgates = qpe.getNindividualOps();

    // extract lambda from QPE global phase
    const auto lambda = qpe.getGlobalPhase();

    iqpe = qc::createIterativeQPE(lambda, precision);
    iqpeNgates = iqpe.getNindividualOps();

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";

    theta = lambda / 2;

    std::cout << "Expected theta=" << theta << "\n";
    std::bitset<64> binaryExpansion{};
    dd::fp expansion = theta * 2;
    std::size_t index = 0;
    while (std::abs(expansion) > 1e-8) {
      if (expansion >= 1.) {
        binaryExpansion.set(index);
        expansion -= 1.0;
      }
      index++;
      expansion *= 2;
    }

    expectedResult = 0ULL;
    for (std::size_t i = 0; i < precision; ++i) {
      if (binaryExpansion.test(i)) {
        expectedResult |= (1ULL << (precision - 1 - i));
      }
    }
    std::stringstream ss{};
    for (auto i = precision; i > 0; --i) {
      if ((expectedResult & (1ULL << (i - 1))) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    expectedResultRepresentation = ss.str();

    secondExpectedResult = expectedResult + 1;
    ss.str("");
    for (auto i = precision; i > 0; --i) {
      if ((secondExpectedResult & (1ULL << (i - 1))) != 0) {
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
};

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

TEST_P(DynamicCircuitEvalInexactQPE, DISABLED_UnitaryTransformation) {
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

class DynamicCircuitEvalBV : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit bitwidth{};
  qc::QuantumComputation bv;
  qc::QuantumComputation dbv;
  std::size_t bvNgates{};
  std::size_t dbvNgates{};
  std::unique_ptr<dd::Package> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    bitwidth = GetParam();

    dd = std::make_unique<dd::Package>(bitwidth + 1);

    bv = qc::createBernsteinVazirani(bitwidth);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(bv);
    bvNgates = bv.getNindividualOps();

    const auto expected = bv.getName().substr(3);
    dbv = qc::createIterativeBernsteinVazirani(qc::BVBitString(expected),
                                               bitwidth);
    dbvNgates = dbv.getNindividualOps();
    std::cout << "Hidden bitstring: " << expected << " (" << bitwidth
              << " qubits)\n";
  }
};

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

TEST_P(DynamicCircuitEvalBV, UnitaryTransformation) {
  bv.reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(dbv);
  qc::CircuitOptimizer::deferMeasurements(dbv);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(dbv);
  dbv.reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  dd::MatrixDD e = dd::Package::makeIdent();
  dd->incRef(e);

  auto leftIt = bv.begin();
  auto rightIt = dbv.begin();

  while (leftIt != bv.end() && rightIt != dbv.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    auto multRight = dd->multiply(multLeft, getInverseDD(**rightIt, *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != bv.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != dbv.end()) {
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
  ss << "bv,transformation," << bv.getNqubits() << "," << bvNgates << ",2,"
     << dbvNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

class DynamicCircuitEvalQFT : public testing::TestWithParam<qc::Qubit> {
protected:
  qc::Qubit precision{};
  qc::QuantumComputation qft;
  qc::QuantumComputation dqft;
  std::size_t qftNgates{};
  std::size_t dqftNgates{};
  std::unique_ptr<dd::Package> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package>(precision);

    qft = qc::createQFT(precision);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(qft);
    qftNgates = qft.getNindividualOps();

    dqft = qc::createIterativeQFT(precision);
    dqftNgates = dqft.getNindividualOps();
  }
};

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

TEST_P(DynamicCircuitEvalQFT, DISABLED_UnitaryTransformation) {
  qft.reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(dqft);
  qc::CircuitOptimizer::deferMeasurements(dqft);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(dqft);
  dqft.reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  dd::MatrixDD e = dd::Package::makeIdent();
  dd->incRef(e);

  auto leftIt = qft.begin();
  auto rightIt = dqft.begin();

  while (leftIt != qft.end() && rightIt != dqft.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    auto multRight = dd->multiply(multLeft, getInverseDD(**rightIt, *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != qft.end()) {
    auto multLeft = dd->multiply(getDD(**leftIt, *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != dqft.end()) {
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
  ss << "qft,transformation," << qft.getNqubits() << "," << qftNgates << ",1,"
     << dqftNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}
