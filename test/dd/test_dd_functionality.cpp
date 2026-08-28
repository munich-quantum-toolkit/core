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
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qasm3/Importer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qc;
using namespace dd;

namespace {

class DDFunctionality : public testing::TestWithParam<OpType> {
protected:
  void TearDown() override {}

  void SetUp() override {
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::ranges::generate(randomData, [&]() { return rd(); });
    std::seed_seq seeds(begin(randomData), end(randomData));
    mt.seed(seeds);
    dist = std::uniform_real_distribution<fp>(0.0, 2. * qc::PI);
  }

  std::mt19937_64 mt;
  std::uniform_real_distribution<fp> dist;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Parameters, DDFunctionality,
    testing::Values(GPhase, I, H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, V, Vdg, U,
                    U2, P, R, RX, RY, RZ, Peres, Peresdg, SWAP, iSWAP, iSWAPdg,
                    DCX, ECR, RXX, RYY, RZZ, RZX, RCCX, XXminusYY, XXplusYY),
    [](const testing::TestParamInfo<DDFunctionality::ParamType>& inf) {
      const auto gate = inf.param;
      return toString(gate);
    });

TEST_P(DDFunctionality, StandardOpBuildInverseBuild) {
  using namespace literals;

  constexpr std::size_t nq = 4;

  const auto dd = std::make_unique<Package>(nq);

  StandardOperation op;
  auto gate = static_cast<OpType>(GetParam());
  switch (gate) {
  case GPhase:
    op = StandardOperation(Controls{}, Targets{}, gate, std::vector{dist(mt)});
    break;
  case U:
    op = StandardOperation(0, gate, std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case U2:
  case R:
    op = StandardOperation(0, gate, std::vector{dist(mt), dist(mt)});
    break;
  case RX:
  case RY:
  case RZ:
  case P:
    op = StandardOperation(0, gate, std::vector{dist(mt)});
    break;
  case SWAP:
  case iSWAP:
  case iSWAPdg:
  case DCX:
  case ECR:
  case Peres:
  case Peresdg:
    op = StandardOperation({}, 0, 1, gate);
    break;
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
    op = StandardOperation(Controls{}, 0, 1, gate, std::vector{dist(mt)});
    break;
  case XXminusYY:
  case XXplusYY:
    op = StandardOperation(Controls{}, 0, 1, gate,
                           std::vector{dist(mt), dist(mt)});
    break;
  case RCCX:
    op = StandardOperation(Targets{0, 1, 2}, gate);
    break;
  default:
    op = StandardOperation(0, gate);
  }

  MatrixDD mDD;
  ASSERT_NO_THROW(
      { mDD = dd->multiply(getDD(op, *dd), getInverseDD(op, *dd)); });
  EXPECT_TRUE(mDD.isIdentity());
}

TEST_P(DDFunctionality, ControlledStandardOpBuildInverseBuild) {
  using namespace literals;

  constexpr std::size_t nq = 4;

  const auto dd = std::make_unique<Package>(nq);

  StandardOperation op;
  auto gate = static_cast<OpType>(GetParam());
  switch (gate) {
  case GPhase:
    op = StandardOperation(Controls{0}, Targets{}, gate, std::vector{dist(mt)});
    break;
  case U:
    op = StandardOperation(0, 1, gate,
                           std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case U2:
  case R:
    op = StandardOperation(0, 1, gate, std::vector{dist(mt), dist(mt)});
    break;
  case RX:
  case RY:
  case RZ:
  case P:
    op = StandardOperation(0, 1, gate, std::vector{dist(mt)});
    break;
  case SWAP:
  case iSWAP:
  case iSWAPdg:
  case DCX:
  case ECR:
  case Peres:
  case Peresdg:
    op = StandardOperation(Controls{0}, 1, 2, gate);
    break;
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
    op = StandardOperation(Controls{0}, 1, 2, gate, std::vector{dist(mt)});
    break;
  case XXminusYY:
  case XXplusYY:
    op = StandardOperation(Controls{0}, 1, 2, gate,
                           std::vector{dist(mt), dist(mt)});
    break;
  case RCCX:
    op = StandardOperation(Controls{0}, Targets{1, 2, 3}, gate);
    break;
  default:
    op = StandardOperation(0, 1, gate);
  }

  MatrixDD mDD;
  ASSERT_NO_THROW(
      { mDD = dd->multiply(getDD(op, *dd), getInverseDD(op, *dd)); });
  EXPECT_TRUE(mDD.isIdentity());
}

TEST_P(DDFunctionality, ControlledStandardNegOpBuildInverseBuild) {
  using namespace literals;

  constexpr std::size_t nq = 4;

  const auto dd = std::make_unique<Package>(nq);

  StandardOperation op;
  auto gate = static_cast<OpType>(GetParam());
  switch (gate) {
  case GPhase:
    op = StandardOperation(Controls{0_nc}, Targets{}, gate,
                           std::vector{dist(mt)});
    break;
  case U:
    op = StandardOperation(Controls{0_nc}, 1, gate,
                           std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case U2:
  case R:
    op = StandardOperation(Controls{0_nc}, 1, gate,
                           std::vector{dist(mt), dist(mt)});
    break;
  case RX:
  case RY:
  case RZ:
  case P:
    op = StandardOperation(Controls{0_nc}, 1, gate, std::vector{dist(mt)});
    break;
  case SWAP:
  case iSWAP:
  case iSWAPdg:
  case DCX:
  case ECR:
  case Peres:
  case Peresdg:
    op = StandardOperation(Controls{0_nc}, 1, 2, gate);
    break;
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
    op = StandardOperation(Controls{0_nc}, 1, 2, gate, std::vector{dist(mt)});
    break;
  case XXminusYY:
  case XXplusYY:
    op = StandardOperation(Controls{0_nc}, 1, 2, gate,
                           std::vector{dist(mt), dist(mt)});
    break;
  case RCCX:
    op = StandardOperation(Controls{0_nc}, Targets{1, 2, 3}, gate);
    break;
  default:
    op = StandardOperation(Controls{0_nc}, 1, gate);
  }

  MatrixDD mDD;
  ASSERT_NO_THROW(
      { mDD = dd->multiply(getDD(op, *dd), getInverseDD(op, *dd)); });
  EXPECT_TRUE(mDD.isIdentity());
}

TEST_F(DDFunctionality, BuildCircuit) {
  constexpr std::size_t nq = 4;

  const auto dd = std::make_unique<Package>(nq);

  QuantumComputation qc(nq);
  qc.x(0);
  qc.swap(0, 1);
  qc.cswap(2, 0, 1);
  qc.mcswap({2, 3}, 0, 1);
  qc.iswap(0, 1);
  qc.ciswap(2, 0, 1);
  qc.mciswap({2, 3}, 0, 1);
  qc.h(0);
  qc.s(3);
  qc.sdg(2);
  qc.v(0);
  qc.t(1);
  qc.cx(0, 1);
  qc.cx(3, 2);
  qc.mcx({2, 3}, 0);
  qc.dcx(0, 1);
  qc.cdcx(2, 0, 1);
  qc.ecr(0, 1);
  qc.cecr(2, 0, 1);
  const auto theta = dist(mt);
  qc.rxx(theta, 0, 1);
  qc.crxx(theta, 2, 0, 1);
  qc.ryy(theta, 0, 1);
  qc.cryy(theta, 2, 0, 1);
  qc.rzz(theta, 0, 1);
  qc.crzz(theta, 2, 0, 1);
  qc.rzx(theta, 0, 1);
  qc.crzx(theta, 2, 0, 1);
  const auto beta = dist(mt);
  qc.xx_minus_yy(theta, beta, 0, 1);
  qc.cxx_minus_yy(theta, beta, 2, 0, 1);
  qc.xx_plus_yy(theta, beta, 0, 1);
  qc.cxx_plus_yy(theta, beta, 2, 0, 1);
  qc.rccx(0, 1, 2);
  qc.crccx(3, 0, 1, 2);
  qc.r(theta, beta, 0);
  qc.cr(theta, beta, 2, 0);
  qc.mcr(theta, beta, {2, 3}, 0);

  // invert the circuit above
  qc.mcr(-theta, beta, {2, 3}, 0);
  qc.cr(-theta, beta, 2, 0);
  qc.r(-theta, beta, 0);
  qc.crccx(3, 0, 1, 2);
  qc.rccx(0, 1, 2);
  qc.cxx_plus_yy(-theta, beta, 2, 0, 1);
  qc.xx_plus_yy(-theta, beta, 0, 1);
  qc.cxx_minus_yy(-theta, beta, 2, 0, 1);
  qc.xx_minus_yy(-theta, beta, 0, 1);
  qc.crzx(-theta, 2, 0, 1);
  qc.rzx(-theta, 0, 1);
  qc.crzz(-theta, 2, 0, 1);
  qc.rzz(-theta, 0, 1);
  qc.cryy(-theta, 2, 0, 1);
  qc.ryy(-theta, 0, 1);
  qc.crxx(-theta, 2, 0, 1);
  qc.rxx(-theta, 0, 1);
  qc.cecr(2, 0, 1);
  qc.ecr(0, 1);
  qc.cdcx(2, 1, 0);
  qc.dcx(1, 0);
  qc.mcx({2, 3}, 0);
  qc.cx(3, 2);
  qc.cx(0, 1);
  qc.tdg(1);
  qc.vdg(0);
  qc.s(2);
  qc.sdg(3);
  qc.h(0);
  qc.mciswapdg({2, 3}, 0, 1);
  qc.ciswapdg(2, 0, 1);
  qc.iswapdg(0, 1);
  qc.mcswap({2, 3}, 0, 1);
  qc.cswap(2, 0, 1);
  qc.swap(0, 1);
  qc.x(0);

  const MatrixDD dd1 = buildFunctionality(qc, *dd);

  qc.x(0);
  const MatrixDD dd2 = buildFunctionality(qc, *dd);

  EXPECT_TRUE(dd1.isIdentity());
  EXPECT_FALSE(dd2.isIdentity());

  dd->decRef(dd1);
  dd->decRef(dd2);
  dd->garbageCollect(true);

  const auto [vector, matrix, reals] = dd->computeActiveCounts();
  EXPECT_EQ(vector, 0);
  EXPECT_EQ(matrix, 0);
  EXPECT_EQ(reals, 0);
}

TEST_F(DDFunctionality, NonUnitary) {
  constexpr std::size_t nq = 4;

  const auto dd = std::make_unique<Package>(nq);

  const QuantumComputation qc{};
  auto dummyMap = Permutation{};
  auto op = NonUnitaryOperation({0, 1, 2, 3}, {0, 1, 2, 3});
  EXPECT_FALSE(op.isUnitary());
  EXPECT_THROW(getDD(op, *dd), std::invalid_argument);
  EXPECT_THROW(getInverseDD(op, *dd), std::invalid_argument);
  EXPECT_THROW(getDD(op, *dd, dummyMap), std::invalid_argument);
  EXPECT_THROW(getInverseDD(op, *dd, dummyMap), std::invalid_argument);
  for (qc::Qubit i = 0; i < nq; ++i) {
    EXPECT_TRUE(op.actsOn(i));
  }

  for (qc::Qubit i = 0; i < nq; ++i) {
    dummyMap[i] = i;
  }
  auto barrier = StandardOperation({0, 1, 2, 3}, OpType::Barrier);
  EXPECT_TRUE(getDD(barrier, *dd).isIdentity());
  EXPECT_TRUE(getInverseDD(barrier, *dd).isIdentity());
  EXPECT_TRUE(getDD(barrier, *dd, dummyMap).isIdentity());
  EXPECT_TRUE(getInverseDD(barrier, *dd, dummyMap).isIdentity());
}

TEST_F(DDFunctionality, CircuitEquivalence) {
  constexpr std::size_t nq = 1;

  const auto dd = std::make_unique<Package>(nq);

  // verify that the IBM decomposition of the H gate into RZ-SX-RZ works as
  // expected (i.e., realizes H up to a global phase)
  QuantumComputation qc1(nq);
  qc1.h(0);

  QuantumComputation qc2(nq);
  qc2.rz(qc::PI_2, 0);
  qc2.sx(0);
  qc2.rz(qc::PI_2, 0);

  const MatrixDD dd1 = buildFunctionality(qc1, *dd);
  const MatrixDD dd2 = buildFunctionality(qc2, *dd);

  EXPECT_EQ(dd1.p, dd2.p);

  dd->decRef(dd1);
  dd->decRef(dd2);
  dd->garbageCollect(true);

  const auto [vector, matrix, reals] = dd->computeActiveCounts();
  EXPECT_EQ(vector, 0);
  EXPECT_EQ(matrix, 0);
  EXPECT_EQ(reals, 0);
}

TEST_F(DDFunctionality, ChangePermutation) {
  const std::string testfile = "// o 1 0\n"
                               "OPENQASM 2.0;"
                               "include \"qelib1.inc\";"
                               "qreg q[2];"
                               "x q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);
  const auto dd = std::make_unique<Package>(qc.getNqubits());

  const auto sim = simulate(qc, makeZeroState(qc.getNqubits(), *dd), *dd);
  EXPECT_TRUE(sim.p->e[0].isZeroTerminal());
  EXPECT_TRUE(sim.p->e[1].w.exactlyOne());
  EXPECT_TRUE(sim.p->e[1].p->e[1].isZeroTerminal());
  EXPECT_TRUE(sim.p->e[1].p->e[0].w.exactlyOne());
  const auto func = buildFunctionality(qc, *dd);
  EXPECT_FALSE(func.p->e[0].isZeroTerminal());
  EXPECT_FALSE(func.p->e[1].isZeroTerminal());
  EXPECT_FALSE(func.p->e[2].isZeroTerminal());
  EXPECT_FALSE(func.p->e[3].isZeroTerminal());
  EXPECT_TRUE(func.p->e[0].p->e[1].w.exactlyOne());
  EXPECT_TRUE(func.p->e[1].p->e[3].w.exactlyOne());
  EXPECT_TRUE(func.p->e[2].p->e[0].w.exactlyOne());
  EXPECT_TRUE(func.p->e[3].p->e[2].w.exactlyOne());
}

TEST_F(DDFunctionality, IfElseOperationConditions) {
  const auto cmpKinds = {ComparisonKind::Eq, ComparisonKind::Neq};
  for (const auto kind : cmpKinds) {
    QuantumComputation qc(1U, 1U);
    // ensure that the state is |1>.
    qc.x(0);
    // measure the qubit to get a classical `1` result to condition on.
    qc.measure(0, 0);
    // apply a classic-controlled X gate whenever the measured result compares
    // as specified by kind with the previously measured result.
    qc.if_(X, 0, 0, true, kind);
    // measure into the same register to check the result.
    qc.measure(0, 0);

    constexpr auto shots = 16U;
    const auto hist = sample(qc, shots);

    EXPECT_EQ(hist.size(), 1);
    const auto& [key, value] = *hist.begin();
    EXPECT_EQ(value, shots);
    if (kind == ComparisonKind::Eq) {
      EXPECT_EQ(key, "0");
    } else {
      EXPECT_EQ(key, "1");
    }
  }
}

TEST_F(DDFunctionality, IfElseOperationElseBranch) {
  QuantumComputation qc(1U, 1U);
  qc.x(0);
  qc.measure(0, 0);
  qc.ifElse(std::make_unique<StandardOperation>(0, I),
            std::make_unique<StandardOperation>(0, X), 0, false);
  qc.measure(0, 0);

  constexpr auto shots = 16U;
  const auto hist = sample(qc, shots);

  EXPECT_EQ(hist.size(), 1);
  const auto& [key, value] = *hist.begin();
  EXPECT_EQ(value, shots);
  EXPECT_EQ(key, "0");
}

TEST_F(DDFunctionality, VectorKroneckerWithTerminal) {
  constexpr std::size_t nq = 1;
  constexpr auto root = vEdge::one();

  const auto dd = std::make_unique<Package>(nq);

  const auto zeroState = makeZeroState(nq, *dd);
  const auto extendedRoot = dd->kronecker(zeroState, root, 0);
  EXPECT_EQ(zeroState, extendedRoot);

  dd->decRef(zeroState);
  dd->garbageCollect(true);

  const auto [vector, matrix, reals] = dd->computeActiveCounts();
  EXPECT_EQ(vector, 0);
  EXPECT_EQ(matrix, 0);
  EXPECT_EQ(reals, 0);
}

TEST_F(DDFunctionality, DynamicCircuitSimulationWithSWAP) {
  QuantumComputation qc(2, 2);
  qc.x(0);
  qc.swap(0, 1);
  qc.measure(1, 0);
  qc.if_(X, 0, 0);
  qc.measure(0, 1);

  constexpr auto shots = 16U;
  const auto hist = sample(qc, shots);
  EXPECT_EQ(hist.size(), 1);
  const auto& [key, value] = *hist.begin();
  EXPECT_EQ(value, shots);
  EXPECT_EQ(key, "11");
}

TEST_F(DDFunctionality, SamplingRetainsReferencedStateWithGlobalPhase) {
  QuantumComputation qc(1U);
  qc.x(0U);
  qc.gphase(qc::PI_2);

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(0U);
  auto result = sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 0U, rng);

  EXPECT_TRUE(result.counts.empty());
  EXPECT_EQ(result.executions, 1U);
  const auto amplitudes = result.state.getVector();
  ASSERT_EQ(amplitudes.size(), 2U);
  EXPECT_NEAR(amplitudes.at(0U).real(), 0., 1e-12);
  EXPECT_NEAR(amplitudes.at(0U).imag(), 0., 1e-12);
  EXPECT_NEAR(amplitudes.at(1U).real(), 0., 1e-12);
  EXPECT_NEAR(amplitudes.at(1U).imag(), 1., 1e-12);

  const auto& roots = dd.getRootSet<vNode>();
  ASSERT_EQ(roots.size(), 1U);
  EXPECT_EQ(roots.at(result.state), 1U);
  EXPECT_NO_THROW(dd.decRef(result.state));
  EXPECT_TRUE(roots.empty());
}

TEST_F(DDFunctionality, SamplingPreservesTerminalMeasurementOrder) {
  QuantumComputation repeatedQubit(1U, 2U);
  repeatedQubit.x(0U);
  repeatedQubit.measure(0U, 0U);
  repeatedQubit.measure(0U, 1U);
  EXPECT_EQ(sample(repeatedQubit, 8U, 1U),
            (std::map<std::string, std::size_t>{{"11", 8U}}));

  QuantumComputation repeatedBit(2U, 1U);
  repeatedBit.x(0U);
  repeatedBit.measure(1U, 0U);
  repeatedBit.measure(0U, 0U);
  EXPECT_EQ(sample(repeatedBit, 8U, 1U),
            (std::map<std::string, std::size_t>{{"1", 8U}}));
}

TEST_F(DDFunctionality, SamplingWithoutMeasurementsUsesQuantumWidth) {
  QuantumComputation qc(2U, 1U);
  qc.x(1U);

  EXPECT_EQ(sample(qc, 8U, 1U),
            (std::map<std::string, std::size_t>{{"10", 8U}}));
}

TEST_F(DDFunctionality, SamplingUsesOutputPermutation) {
  QuantumComputation qc(2U);
  qc.x(0U);
  qc.outputPermutation = {{0U, 1U}, {1U, 0U}};

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(17U);
  auto result = sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 8U, rng);

  EXPECT_EQ(result.counts, (std::map<std::string, std::size_t>{{"10", 8U}}));
  const auto amplitudes = result.state.getVector();
  ASSERT_EQ(amplitudes.size(), 4U);
  EXPECT_NEAR(std::abs(amplitudes.at(2U)), 1., 1e-12);
  dd.decRef(result.state);
}

TEST_F(DDFunctionality, SamplingRetainsCanonicalStateAcrossVirtualSwap) {
  QuantumComputation qc(2U, 2U);
  qc.x(0U);
  qc.swap(0U, 1U);
  qc.measure(0U, 0U);
  qc.measure(1U, 1U);

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(17U);
  auto result = sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 8U, rng);

  EXPECT_EQ(result.counts, (std::map<std::string, std::size_t>{{"10", 8U}}));
  EXPECT_EQ(result.executions, 1U);
  const auto amplitudes = result.state.getVector();
  ASSERT_EQ(amplitudes.size(), 4U);
  EXPECT_NEAR(std::abs(amplitudes.at(2U)), 1., 1e-12);
  dd.decRef(result.state);
}

TEST_F(DDFunctionality, SamplingReducesGarbageInCountsAndRetainedState) {
  QuantumComputation qc(3U);
  qc.h(0U);
  qc.cx(0U, 2U);
  qc.setLogicalQubitGarbage(2U);
  qc.outputPermutation = {{0U, 1U}, {1U, 0U}};

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(17U);
  auto result = sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 32U, rng);

  EXPECT_EQ(result.executions, 1U);
  EXPECT_EQ(result.counts.size(), 2U);
  EXPECT_EQ(result.counts.at("000") + result.counts.at("010"), 32U);
  const auto amplitudes = result.state.getVector();
  ASSERT_EQ(amplitudes.size(), 8U);
  EXPECT_NEAR(std::abs(amplitudes.at(0U)), dd::SQRT2_2, 1e-12);
  EXPECT_NEAR(std::abs(amplitudes.at(2U)), dd::SQRT2_2, 1e-12);
  for (const auto index : {1U, 3U, 4U, 5U, 6U, 7U}) {
    EXPECT_NEAR(std::abs(amplitudes.at(index)), 0., 1e-12);
  }
  dd.decRef(result.state);
}

TEST_F(DDFunctionality, DynamicSamplingRetainsLastStateAndBalancesRoots) {
  QuantumComputation qc(1U, 1U);
  qc.x(0U);
  qc.measure(0U, 0U);
  qc.reset(0U);

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(17U);
  auto result = sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 8U, rng);

  EXPECT_EQ(result.counts, (std::map<std::string, std::size_t>{{"1", 8U}}));
  EXPECT_EQ(result.executions, 8U);
  const auto amplitudes = result.state.getVector();
  ASSERT_EQ(amplitudes.size(), 2U);
  EXPECT_NEAR(std::abs(amplitudes.at(0U)), 1., 1e-12);
  EXPECT_NEAR(std::abs(amplitudes.at(1U)), 0., 1e-12);
  const auto& roots = dd.getRootSet<vNode>();
  ASSERT_EQ(roots.size(), 1U);
  EXPECT_EQ(roots.at(result.state), 1U);
  dd.decRef(result.state);
  EXPECT_TRUE(roots.empty());
}

TEST_F(DDFunctionality, DynamicSamplingReusesCallerRandomNumberGenerator) {
  QuantumComputation qc(1U, 1U);
  qc.h(0U);
  qc.measure(0U, 0U);
  qc.x(0U);

  Package splitPackage(qc.getNqubits());
  std::mt19937_64 splitRng(0U);
  auto first = sample(qc, makeZeroState(qc.getNqubits(), splitPackage),
                      splitPackage, 17U, splitRng);
  auto second = sample(qc, makeZeroState(qc.getNqubits(), splitPackage),
                       splitPackage, 23U, splitRng);
  auto splitCounts = first.counts;
  for (const auto& [state, count] : second.counts) {
    splitCounts[state] += count;
  }

  Package combinedPackage(qc.getNqubits());
  std::mt19937_64 combinedRng(0U);
  auto combined = sample(qc, makeZeroState(qc.getNqubits(), combinedPackage),
                         combinedPackage, 40U, combinedRng);

  EXPECT_EQ(splitCounts, combined.counts);
  EXPECT_EQ(splitRng, combinedRng);
  EXPECT_EQ(first.executions, 17U);
  EXPECT_EQ(second.executions, 23U);
  EXPECT_EQ(combined.executions, 40U);

  splitPackage.decRef(first.state);
  splitPackage.decRef(second.state);
  combinedPackage.decRef(combined.state);
}

TEST_F(DDFunctionality, ResetWithoutMeasurementsUsesQuantumWidth) {
  QuantumComputation qc(2U, 1U);
  qc.x(1U);
  qc.reset(0U);

  EXPECT_EQ(sample(qc, 8U, 1U),
            (std::map<std::string, std::size_t>{{"10", 8U}}));
}

TEST_F(DDFunctionality, PackageAwareSampleConsumesDynamicInput) {
  QuantumComputation qc(1U, 1U);
  qc.h(0U);
  qc.measure(0U, 0U);
  qc.x(0U);

  Package dd(qc.getNqubits());
  EXPECT_EQ(sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 8U, 17U).size(),
            2U);
  EXPECT_TRUE(dd.getRootSet<vNode>().empty());
}

TEST_F(DDFunctionality, SamplingReleasesTransferredInputOnAnalysisFailure) {
  QuantumComputation qc(1U, 1U);
  qc.measure(0U, 0U);
  auto& measurement = dynamic_cast<NonUnitaryOperation&>(*qc.at(0U));
  measurement.getClassics().clear();

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(17U);
  EXPECT_THROW(static_cast<void>(
                   sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 1U, rng)),
               std::invalid_argument);
  EXPECT_TRUE(dd.getRootSet<vNode>().empty());
}

TEST_F(DDFunctionality,
       SamplingReleasesTransferredInputOnInvalidMeasurementBit) {
  QuantumComputation qc(1U, 1U);
  qc.h(0U);
  qc.measure(0U, 0U);
  qc.x(0U);
  auto& measurement = dynamic_cast<NonUnitaryOperation&>(*qc.at(1U));
  measurement.getClassics().at(0U) = 1U;

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(17U);
  EXPECT_THROW(static_cast<void>(
                   sample(qc, makeZeroState(qc.getNqubits(), dd), dd, 1U, rng)),
               std::out_of_range);
  EXPECT_TRUE(dd.getRootSet<vNode>().empty());
}

TEST_F(DDFunctionality, DynamicSamplingZeroShotsRetainsInput) {
  QuantumComputation qc(1U, 1U);
  qc.measure(0U, 0U);
  qc.x(0U);

  Package dd(qc.getNqubits());
  std::mt19937_64 rng(0U);
  const auto input = makeZeroState(qc.getNqubits(), dd);
  auto result = sample(qc, input, dd, 0U, rng);

  EXPECT_TRUE(result.counts.empty());
  EXPECT_EQ(result.executions, 0U);
  EXPECT_EQ(result.state, input);
  const auto& roots = dd.getRootSet<vNode>();
  ASSERT_EQ(roots.size(), 1U);
  EXPECT_EQ(roots.at(result.state), 1U);
  dd.decRef(result.state);
}

TEST_F(DDFunctionality, SamplingReusesCallerRandomNumberGenerator) {
  QuantumComputation qc(1U);
  qc.h(0U);

  Package splitPackage(qc.getNqubits());
  std::mt19937_64 splitRng(0U);
  auto first = sample(qc, makeZeroState(qc.getNqubits(), splitPackage),
                      splitPackage, 17U, splitRng);
  auto second = sample(qc, makeZeroState(qc.getNqubits(), splitPackage),
                       splitPackage, 23U, splitRng);
  auto splitCounts = first.counts;
  for (const auto& [state, count] : second.counts) {
    splitCounts[state] += count;
  }

  Package combinedPackage(qc.getNqubits());
  std::mt19937_64 combinedRng(0U);
  auto combined = sample(qc, makeZeroState(qc.getNqubits(), combinedPackage),
                         combinedPackage, 40U, combinedRng);

  EXPECT_EQ(splitCounts, combined.counts);
  EXPECT_EQ(splitRng, combinedRng);
  EXPECT_EQ(first.executions, 1U);
  EXPECT_EQ(second.executions, 1U);
  EXPECT_EQ(combined.executions, 1U);

  splitPackage.decRef(first.state);
  splitPackage.decRef(second.state);
  combinedPackage.decRef(combined.state);
}
