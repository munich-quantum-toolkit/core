/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qasm3/Importer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qc;
using namespace dd;

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

INSTANTIATE_TEST_SUITE_P(
    Parameters, DDFunctionality,
    testing::Values(GPhase, I, H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, V, Vdg, U,
                    U2, P, RX, RY, RZ, Peres, Peresdg, SWAP, iSWAP, iSWAPdg,
                    DCX, ECR, RXX, RYY, RZZ, RZX, XXminusYY, XXplusYY),
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

  // invert the circuit above
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

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
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

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
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

TEST_F(DDFunctionality, FuseTwoSingleQubitGates) {
  constexpr std::size_t nq = 1;

  const auto dd = std::make_unique<Package>(nq);

  QuantumComputation qc(nq);
  qc.x(0);
  qc.h(0);

  qc.print(std::cout);
  const MatrixDD baseDD = buildFunctionality(qc, *dd);

  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto optDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(baseDD, optDD);

  dd->decRef(baseDD);
  dd->decRef(optDD);
  dd->garbageCollect(true);

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
}

TEST_F(DDFunctionality, FuseThreeSingleQubitGates) {
  constexpr std::size_t nq = 1;

  const auto dd = std::make_unique<Package>(nq);

  QuantumComputation qc(nq);
  qc.x(0);
  qc.h(0);
  qc.y(0);

  const MatrixDD baseDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  CircuitOptimizer::singleQubitGateFusion(qc);
  const MatrixDD optDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(baseDD, optDD);

  dd->decRef(baseDD);
  dd->decRef(optDD);
  dd->garbageCollect(true);

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
}

TEST_F(DDFunctionality, FuseNoSingleQubitGates) {
  constexpr std::size_t nq = 2;

  const auto dd = std::make_unique<Package>(nq);

  QuantumComputation qc(nq);
  qc.h(0);
  qc.cx(0, 1);
  qc.y(0);

  const MatrixDD baseDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  CircuitOptimizer::singleQubitGateFusion(qc);
  const MatrixDD optDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  EXPECT_EQ(qc.getNops(), 3);
  EXPECT_EQ(baseDD, optDD);

  dd->decRef(baseDD);
  dd->decRef(optDD);
  dd->garbageCollect(true);

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
}

TEST_F(DDFunctionality, FuseSingleQubitGatesAcrossOtherGates) {
  constexpr std::size_t nq = 2;

  const auto dd = std::make_unique<Package>(nq);

  QuantumComputation qc(nq);
  qc.h(0);
  qc.z(1);
  qc.y(0);
  const MatrixDD baseDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto optDD = buildFunctionality(qc, *dd);

  std::cout << "-----------------------------\n";
  qc.print(std::cout);

  EXPECT_EQ(qc.getNops(), 2);
  EXPECT_EQ(baseDD, optDD);

  dd->decRef(baseDD);
  dd->decRef(optDD);
  dd->garbageCollect(true);

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
}

TEST_F(DDFunctionality, ClassicControlledOperationConditions) {
  const auto cmpKinds = {ComparisonKind::Eq, ComparisonKind::Neq};
  for (const auto kind : cmpKinds) {
    QuantumComputation qc(1U, 1U);
    // ensure that the state is |1>.
    qc.x(0);
    // measure the qubit to get a classical `1` result to condition on.
    qc.measure(0, 0);
    // apply a classic-controlled X gate whenever the measured result compares
    // as specified by kind with the previously measured result.
    qc.classicControlled(X, 0, 0, 1U, kind);
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

TEST_F(DDFunctionality, VectorKroneckerWithTerminal) {
  constexpr std::size_t nq = 1;
  constexpr auto root = vEdge::one();

  const auto dd = std::make_unique<Package>(nq);

  const auto zeroState = makeZeroState(nq, *dd);
  const auto extendedRoot = dd->kronecker(zeroState, root, 0);
  EXPECT_EQ(zeroState, extendedRoot);

  dd->decRef(zeroState);
  dd->garbageCollect(true);

  const auto counts = dd->computeActiveCounts();
  EXPECT_EQ(counts.vector, 0);
  EXPECT_EQ(counts.density, 0);
  EXPECT_EQ(counts.matrix, 0);
  EXPECT_EQ(counts.reals, 0);
}

TEST_F(DDFunctionality, DynamicCircuitSimulationWithSWAP) {
  QuantumComputation qc(2, 2);
  qc.x(0);
  qc.swap(0, 1);
  qc.measure(1, 0);
  qc.classicControlled(X, 0, 0);
  qc.measure(0, 1);

  constexpr auto shots = 16U;
  const auto hist = sample(qc, shots);
  EXPECT_EQ(hist.size(), 1);
  const auto& [key, value] = *hist.begin();
  EXPECT_EQ(value, shots);
  EXPECT_EQ(key, "11");
}
