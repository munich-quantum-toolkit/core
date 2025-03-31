/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/QFT.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/Simulation.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>

class QFT : public testing::TestWithParam<qc::Qubit> {
protected:
  void TearDown() override {}

  void SetUp() override {
    nqubits = GetParam();
    dd = std::make_unique<dd::Package<>>(nqubits);
  }

  qc::Qubit nqubits = 0;
  std::unique_ptr<dd::Package<>> dd;
  qc::QuantumComputation qc;
  qc::VectorDD sim{};
  qc::MatrixDD func{};
};

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
///	Utilizing more qubits requires the use of fp=long double
constexpr qc::Qubit QFT_MAX_QUBITS = 17U;

constexpr size_t INITIAL_COMPLEX_COUNT = 1;

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

TEST_P(QFT, Functionality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = qc::createQFT(nqubits, false); });
  // there should be no error building the functionality

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionality(qc, *dd); });

  qc.printStatistics(std::cout);
  // QFT DD should consist of 2^n nodes
  ASSERT_EQ(func.size(), 1ULL << nqubits);

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // the final DD should store all 2^n different amplitudes
  // since only positive real values are stored in the complex table
  // this number has to be divided by 4
  ASSERT_EQ(dd->cn.realCount(),
            1ULL << (std::max<std::size_t>(2UL, nqubits) - 2));

  // top edge weight should equal sqrt(0.5)^n
  EXPECT_NEAR(dd::RealNumber::val(func.w.r),
              static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
              dd::RealNumber::eps);

  // first row and first column should consist only of (1/sqrt(2))**nqubits
  for (std::uint64_t i = 0; i < 1ULL << nqubits; ++i) {
    auto c = func.getValueByIndex(dd->qubits(), 0, i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
    c = func.getValueByIndex(dd->qubits(), i, 0);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
  }
  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, FunctionalityRecursive) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = qc::createQFT(nqubits, false); });

  // there should be no error building the functionality
  ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc, *dd); });

  qc.printStatistics(std::cout);
  // QFT DD should consist of 2^n nodes
  ASSERT_EQ(func.size(), 1ULL << nqubits);

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // the final DD should store all 2^n different amplitudes
  // since only positive real values are stored in the complex table
  // this number has to be divided by 4
  ASSERT_EQ(dd->cn.realCount(),
            1ULL << (std::max<std::size_t>(2UL, nqubits) - 2));

  // top edge weight should equal sqrt(0.5)^n
  EXPECT_NEAR(dd::RealNumber::val(func.w.r),
              static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
              dd::RealNumber::eps);

  // first row and first column should consist only of (1/sqrt(2))**nqubits
  for (std::uint64_t i = 0; i < std::pow(static_cast<long double>(2), nqubits);
       ++i) {
    auto c = func.getValueByIndex(dd->qubits(), 0, i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
    c = func.getValueByIndex(dd->qubits(), i, 0);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
  }
  dd->decRef(func);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, Simulation) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = qc::createQFT(nqubits, false); });

  // there should be no error simulating the circuit
  ASSERT_NO_THROW({
    auto in = dd->makeZeroState(nqubits);
    sim = simulate(qc, in, *dd);
  });
  qc.printStatistics(std::cout);

  // QFT DD |0...0> sim should consist of n nodes
  ASSERT_EQ(sim.size(), nqubits + 1);

  // Force garbage collection of compute table and complex table
  dd->garbageCollect(true);

  // top edge weight should equal 1
  EXPECT_NEAR(dd::RealNumber::val(sim.w.r), 1, dd::RealNumber::eps);
  EXPECT_NEAR(dd::RealNumber::val(sim.w.i), 0, dd::RealNumber::eps);

  // first column should consist only of sqrt(0.5)^n's
  for (std::uint64_t i = 0; i < 1ULL << nqubits; ++i) {
    auto c = sim.getValueByIndex(i);
    EXPECT_NEAR(c.real(),
                static_cast<dd::fp>(std::pow(1.L / std::sqrt(2.L), nqubits)),
                dd::RealNumber::eps);
    EXPECT_NEAR(c.imag(), 0, dd::RealNumber::eps);
  }
  dd->decRef(sim);
  dd->garbageCollect(true);
  // number of complex table entries after clean-up should equal initial
  // number of entries
  EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
}

TEST_P(QFT, FunctionalityRecursiveEquality) {
  // there should be no error constructing the circuit
  ASSERT_NO_THROW({ qc = qc::createQFT(nqubits, false); });

  // there should be no error building the functionality recursively
  ASSERT_NO_THROW({ func = buildFunctionalityRecursive(qc, *dd); });

  // there should be no error building the functionality regularly
  qc::MatrixDD funcRec{};
  ASSERT_NO_THROW({ funcRec = buildFunctionality(qc, *dd); });

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
    dd->garbageCollect(true);
    // number of complex table entries after clean-up should equal initial
    // number of entries
    EXPECT_EQ(dd->cn.realCount(), INITIAL_COMPLEX_COUNT);
  }
}
