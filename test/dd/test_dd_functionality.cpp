/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
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

class DDFunctionality : public testing::TestWithParam<qc::OpType> {
protected:
  void TearDown() override {
    if (!e.isTerminal()) {
      dd->decRef(e);
    }
    dd->garbageCollect(true);

    // number of complex table entries after clean-up should equal initial
    // number of entries
    EXPECT_EQ(dd->cn.realCount(), initialComplexCount);
  }

  void SetUp() override {
    // dd
    dd = std::make_unique<dd::Package<>>(nqubits);
    initialComplexCount = dd->cn.realCount();

    // initial state preparation
    e = ident = dd->makeIdent();
    dd->incRef(ident);

    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::generate(begin(randomData), end(randomData), [&]() { return rd(); });
    std::seed_seq seeds(begin(randomData), end(randomData));
    mt.seed(seeds);
    dist = std::uniform_real_distribution<dd::fp>(0.0, 2. * dd::PI);
  }

  std::size_t nqubits = 4U;
  std::size_t initialComplexCount = 0U;
  qc::MatrixDD e{}, ident{};
  std::unique_ptr<dd::Package<>> dd;
  std::mt19937_64 mt;
  std::uniform_real_distribution<dd::fp> dist;
};

INSTANTIATE_TEST_SUITE_P(
    Parameters, DDFunctionality,
    testing::Values(qc::GPhase, qc::I, qc::H, qc::X, qc::Y, qc::Z, qc::S,
                    qc::Sdg, qc::T, qc::Tdg, qc::SX, qc::SXdg, qc::V, qc::Vdg,
                    qc::U, qc::U2, qc::P, qc::RX, qc::RY, qc::RZ, qc::Peres,
                    qc::Peresdg, qc::SWAP, qc::iSWAP, qc::iSWAPdg, qc::DCX,
                    qc::ECR, qc::RXX, qc::RYY, qc::RZZ, qc::RZX, qc::XXminusYY,
                    qc::XXplusYY),
    [](const testing::TestParamInfo<DDFunctionality::ParamType>& inf) {
      const auto gate = inf.param;
      return toString(gate);
    });

TEST_P(DDFunctionality, standardOpBuildInverseBuild) {
  using namespace qc::literals;
  auto gate = static_cast<qc::OpType>(GetParam());

  qc::StandardOperation op;
  switch (gate) {
  case qc::GPhase:
    op = qc::StandardOperation(Controls{}, Targets{}, gate,
                               std::vector{dist(mt)});
    break;
  case qc::U:
    op = qc::StandardOperation(0, gate,
                               std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case qc::U2:
    op = qc::StandardOperation(0, gate, std::vector{dist(mt), dist(mt)});
    break;
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::P:
    op = qc::StandardOperation(0, gate, std::vector{dist(mt)});
    break;

  case qc::SWAP:
  case qc::iSWAP:
  case qc::iSWAPdg:
  case qc::DCX:
  case qc::ECR:
  case qc::Peres:
  case qc::Peresdg:
    op = qc::StandardOperation({}, 0, 1, gate);
    break;
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
    op = qc::StandardOperation(Controls{}, 0, 1, gate, std::vector{dist(mt)});
    break;
  case qc::XXminusYY:
  case qc::XXplusYY:
    op = qc::StandardOperation(Controls{}, 0, 1, gate,
                               std::vector{dist(mt), dist(mt)});
    break;
  default:
    op = qc::StandardOperation(0, gate);
  }

  ASSERT_NO_THROW({ e = dd->multiply(getDD(op, *dd), getInverseDD(op, *dd)); });
  dd->incRef(e);

  EXPECT_EQ(ident, e);
}

TEST_P(DDFunctionality, controlledStandardOpBuildInverseBuild) {
  using namespace qc::literals;
  auto gate = static_cast<qc::OpType>(GetParam());

  qc::StandardOperation op;
  switch (gate) {
  case qc::GPhase:
    op = qc::StandardOperation(Controls{0}, Targets{}, gate,
                               std::vector{dist(mt)});
    break;
  case qc::U:
    op = qc::StandardOperation(0, 1, gate,
                               std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case qc::U2:
    op = qc::StandardOperation(0, 1, gate, std::vector{dist(mt), dist(mt)});
    break;
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::P:
    op = qc::StandardOperation(0, 1, gate, std::vector{dist(mt)});
    break;

  case qc::SWAP:
  case qc::iSWAP:
  case qc::iSWAPdg:
  case qc::DCX:
  case qc::ECR:
  case qc::Peres:
  case qc::Peresdg:
    op = qc::StandardOperation(Controls{0}, 1, 2, gate);
    break;
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
    op = qc::StandardOperation(Controls{0}, 1, 2, gate, std::vector{dist(mt)});
    break;
  case qc::XXminusYY:
  case qc::XXplusYY:
    op = qc::StandardOperation(Controls{0}, 1, 2, gate,
                               std::vector{dist(mt), dist(mt)});
    break;
  default:
    op = qc::StandardOperation(0, 1, gate);
  }

  ASSERT_NO_THROW({ e = dd->multiply(getDD(op, *dd), getInverseDD(op, *dd)); });
  dd->incRef(e);

  EXPECT_EQ(ident, e);
}

TEST_P(DDFunctionality, controlledStandardNegOpBuildInverseBuild) {
  using namespace qc::literals;
  auto gate = static_cast<qc::OpType>(GetParam());

  qc::StandardOperation op;
  switch (gate) {
  case qc::GPhase:
    op = qc::StandardOperation(Controls{0_nc}, Targets{}, gate,
                               std::vector{dist(mt)});
    break;
  case qc::U:
    op = qc::StandardOperation(Controls{0_nc}, 1, gate,
                               std::vector{dist(mt), dist(mt), dist(mt)});
    break;
  case qc::U2:
    op = qc::StandardOperation(Controls{0_nc}, 1, gate,
                               std::vector{dist(mt), dist(mt)});
    break;
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::P:
    op = qc::StandardOperation(Controls{0_nc}, 1, gate, std::vector{dist(mt)});
    break;

  case qc::SWAP:
  case qc::iSWAP:
  case qc::iSWAPdg:
  case qc::DCX:
  case qc::ECR:
  case qc::Peres:
  case qc::Peresdg:
    op = qc::StandardOperation(Controls{0_nc}, 1, 2, gate);
    break;
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
    op = qc::StandardOperation(Controls{0_nc}, 1, 2, gate,
                               std::vector{dist(mt)});
    break;
  case qc::XXminusYY:
  case qc::XXplusYY:
    op = qc::StandardOperation(Controls{0_nc}, 1, 2, gate,
                               std::vector{dist(mt), dist(mt)});
    break;
  default:
    op = qc::StandardOperation(Controls{0_nc}, 1, gate);
  }

  ASSERT_NO_THROW({ e = dd->multiply(getDD(op, *dd), getInverseDD(op, *dd)); });
  dd->incRef(e);

  EXPECT_EQ(ident, e);
}

TEST_F(DDFunctionality, buildCircuit) {
  qc::QuantumComputation qc(nqubits);

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

  e = buildFunctionality(qc, *dd);
  EXPECT_EQ(ident, e);

  qc.x(0);
  e = buildFunctionality(qc, *dd);
  dd->incRef(e);
  EXPECT_NE(ident, e);
}

TEST_F(DDFunctionality, nonUnitary) {
  const qc::QuantumComputation qc{};
  auto dummyMap = Permutation{};
  auto op = qc::NonUnitaryOperation({0, 1, 2, 3}, {0, 1, 2, 3});
  EXPECT_FALSE(op.isUnitary());
  EXPECT_THROW(getDD(op, *dd), std::invalid_argument);
  EXPECT_THROW(getInverseDD(op, *dd), std::invalid_argument);
  EXPECT_THROW(getDD(op, *dd, dummyMap), std::invalid_argument);
  EXPECT_THROW(getInverseDD(op, *dd, dummyMap), std::invalid_argument);
  for (Qubit i = 0; i < nqubits; ++i) {
    EXPECT_TRUE(op.actsOn(i));
  }

  for (Qubit i = 0; i < nqubits; ++i) {
    dummyMap[i] = i;
  }
  auto barrier = qc::StandardOperation({0, 1, 2, 3}, qc::OpType::Barrier);
  EXPECT_TRUE(getDD(barrier, *dd).isIdentity());
  EXPECT_TRUE(getInverseDD(barrier, *dd).isIdentity());
  EXPECT_TRUE(getDD(barrier, *dd, dummyMap).isIdentity());
  EXPECT_TRUE(getInverseDD(barrier, *dd, dummyMap).isIdentity());
}

TEST_F(DDFunctionality, CircuitEquivalence) {
  // verify that the IBM decomposition of the H gate into RZ-SX-RZ works as
  // expected (i.e., realizes H up to a global phase)
  qc::QuantumComputation qc1(1);
  qc1.h(0);

  qc::QuantumComputation qc2(1);
  qc2.rz(PI_2, 0);
  qc2.sx(0);
  qc2.rz(PI_2, 0);

  const qc::MatrixDD dd1 = buildFunctionality(qc1, *dd);
  const qc::MatrixDD dd2 = buildFunctionality(qc2, *dd);

  EXPECT_EQ(dd1.p, dd2.p);
}

TEST_F(DDFunctionality, changePermutation) {
  const std::string testfile = "// o 1 0\n"
                               "OPENQASM 2.0;"
                               "include \"qelib1.inc\";"
                               "qreg q[2];"
                               "x q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);
  const auto sim = simulate(qc, dd->makeZeroState(qc.getNqubits()), *dd);
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
  nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.h(0);

  qc.print(std::cout);
  e = buildFunctionality(qc, *dd);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseThreeSingleQubitGates) {
  nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.h(0);
  qc.y(0);

  e = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseNoSingleQubitGates) {
  nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.h(0);
  qc.cx(0, 1);
  qc.y(0);
  e = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 3);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseSingleQubitGatesAcrossOtherGates) {
  nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.h(0);
  qc.z(1);
  qc.y(0);
  e = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  const auto f = buildFunctionality(qc, *dd);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
  EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, classicControlledOperationConditions) {
  const auto cmpKinds = {ComparisonKind::Eq, ComparisonKind::Neq};
  for (const auto kind : cmpKinds) {
    QuantumComputation qc(1U, 1U);
    // ensure that the state is |1>.
    qc.x(0);
    // measure the qubit to get a classical `1` result to condition on.
    qc.measure(0, 0);
    // apply a classic-controlled X gate whenever the measured result compares
    // as specified by kind with the previously measured result.
    qc.classicControlled(qc::X, 0, 0, 1U, kind);
    // measure into the same register to check the result.
    qc.measure(0, 0);

    constexpr auto shots = 16U;
    const auto hist = dd::sample(qc, shots);

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

TEST_F(DDFunctionality, vectorKroneckerWithTerminal) {
  const auto root = dd::vEdge::one();
  const auto zeroState = dd->makeZeroState(1);
  const auto extendedRoot = dd->kronecker(zeroState, root, 0);
  EXPECT_EQ(zeroState, extendedRoot);
}

TEST_F(DDFunctionality, dynamicCircuitSimulationWithSWAP) {
  QuantumComputation qc(2, 2);
  qc.x(0);
  qc.swap(0, 1);
  qc.measure(1, 0);
  qc.classicControlled(qc::X, 0, 0);
  qc.measure(0, 1);

  constexpr auto shots = 16U;
  const auto hist = dd::sample(qc, shots);
  EXPECT_EQ(hist.size(), 1);
  const auto& [key, value] = *hist.begin();
  EXPECT_EQ(value, shots);
  EXPECT_EQ(key, "11");
}
