/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"

#include <gtest/gtest.h>
#include "ir/operations/SymbolicOperation.hpp"
#include <memory>
#include <stdexcept>
#include <tuple>

using namespace qc;
using namespace sym;
class SymbolicTest : public ::testing::Test {
public:
  Variable x = Variable("x");
  Variable y = Variable("y");
  Variable z = Variable("z");

  Symbolic xMonom = Symbolic{Term<fp>{x}};
  Symbolic yMonom = Symbolic{Term<fp>{y}};
  Symbolic zMonom = Symbolic{Term<fp>{z}};

  QuantumComputation symQc = QuantumComputation(4);
  QuantumComputation qc = QuantumComputation(4);
};

TEST_F(SymbolicTest, Gates) {
  auto xVal = PI_4 / 2;
  auto yVal = PI_4 / 4;
  auto zVal = PI / 3;

  // test all kinds of symbolic operations supported
  symQc.mcu(xMonom, yMonom, zMonom, {1, 2_nc}, 0);
  symQc.cu(xMonom, yMonom, zMonom, 1, 0);
  symQc.u(xMonom, yMonom, zMonom, 0);

  symQc.mcu2(xMonom, yMonom, {1, 2_nc}, 0);
  symQc.cu2(xMonom, yMonom, 1, 0);
  symQc.u2(xMonom, yMonom, 0);

  symQc.mcp(xMonom, {1, 2_nc}, 0);
  symQc.cp(xMonom, 1, 0);
  symQc.p(xMonom, 0);

  symQc.mcrx(xMonom, {1, 2_nc}, 0);
  symQc.crx(xMonom, 1, 0);
  symQc.rx(xMonom, 0);

  symQc.mcry(xMonom, {1, 2_nc}, 0);
  symQc.cry(xMonom, 1, 0);
  symQc.ry(xMonom, 0);

  symQc.mcrz(xMonom, {1, 2_nc}, 0);
  symQc.crz(xMonom, 1, 0);
  symQc.rz(xMonom, 0);

  symQc.mcrxx(xMonom, {3, 2_nc}, 0, 1);
  symQc.crxx(xMonom, 2, 0, 1);
  symQc.rxx(xMonom, 0, 1);

  symQc.mcryy(xMonom, {3, 2_nc}, 0, 1);
  symQc.cryy(xMonom, 2, 0, 1);
  symQc.ryy(xMonom, 0, 1);

  symQc.mcrzz(xMonom, {3, 2_nc}, 0, 1);
  symQc.crzz(xMonom, 2, 0, 1);
  symQc.rzz(xMonom, 0, 1);

  symQc.mcrzx(xMonom, {3, 2_nc}, 0, 1);
  symQc.crzx(xMonom, 2, 0, 1);
  symQc.rzx(xMonom, 0, 1);

  symQc.mcxx_minus_yy(xMonom, yMonom, {3, 2_nc}, 0, 1);
  symQc.cxx_minus_yy(xMonom, yMonom, 2, 0, 1);
  symQc.xx_minus_yy(xMonom, yMonom, 0, 1);

  symQc.mcxx_plus_yy(xMonom, yMonom, {3, 2_nc}, 0, 1);
  symQc.cxx_plus_yy(xMonom, yMonom, 2, 0, 1);
  symQc.xx_plus_yy(xMonom, yMonom, 0, 1);

  EXPECT_FALSE(symQc.isVariableFree());
  for (const auto& symOp : symQc) {
    EXPECT_TRUE(symOp->isSymbolicOperation());
  }

  // normal circuit
  qc.mcu(xVal, yVal, zVal, {1, 2_nc}, 0);
  qc.cu(xVal, yVal, zVal, 1, 0);
  qc.u(xVal, yVal, zVal, 0);

  qc.mcu2(xVal, yVal, {1, 2_nc}, 0);
  qc.cu2(xVal, yVal, 1, 0);
  qc.u2(xVal, yVal, 0);

  qc.mcp(xVal, {1, 2_nc}, 0);
  qc.cp(xVal, 1, 0);
  qc.p(xVal, 0);

  qc.mcrx(xVal, {1, 2_nc}, 0);
  qc.crx(xVal, 1, 0);
  qc.rx(xVal, 0);

  qc.mcry(xVal, {1, 2_nc}, 0);
  qc.cry(xVal, 1, 0);
  qc.ry(xVal, 0);

  qc.mcrz(xVal, {1, 2_nc}, 0);
  qc.crz(xVal, 1, 0);
  qc.rz(xVal, 0);

  qc.mcrxx(xVal, {3, 2_nc}, 0, 1);
  qc.crxx(xVal, 2, 0, 1);
  qc.rxx(xVal, 0, 1);

  qc.mcryy(xVal, {3, 2_nc}, 0, 1);
  qc.cryy(xVal, 2, 0, 1);
  qc.ryy(xVal, 0, 1);

  qc.mcrzz(xVal, {3, 2_nc}, 0, 1);
  qc.crzz(xVal, 2, 0, 1);
  qc.rzz(xVal, 0, 1);

  qc.mcrzx(xVal, {3, 2_nc}, 0, 1);
  qc.crzx(xVal, 2, 0, 1);
  qc.rzx(xVal, 0, 1);

  qc.mcxx_minus_yy(xVal, yVal, {3, 2_nc}, 0, 1);
  qc.cxx_minus_yy(xVal, yVal, 2, 0, 1);
  qc.xx_minus_yy(xVal, yVal, 0, 1);

  qc.mcxx_plus_yy(xVal, yVal, {3, 2_nc}, 0, 1);
  qc.cxx_plus_yy(xVal, yVal, 2, 0, 1);
  qc.xx_plus_yy(xVal, yVal, 0, 1);

  EXPECT_TRUE(qc.isVariableFree());

  // no operation in the uninstantiated circuit should be equal to the standard
  // circuit
  for (auto it1 = symQc.begin(), it2 = qc.begin();
       it1 != symQc.end() && it2 != qc.end(); ++it1, ++it2) {
    EXPECT_FALSE((*it1)->equals(*(*it2)));
  }

  const VariableAssignment assignment{{x, xVal}, {y, yVal}, {z, zVal}};
  auto instantiated = symQc.instantiate(assignment);

  // after the instantiation, the symbolic circuit should be equal to the
  // standard circuit
  for (auto it1 = instantiated.begin(), it2 = qc.begin();
       it1 != instantiated.end() && it2 != qc.end(); ++it1, ++it2) {
    EXPECT_TRUE((*it1)->equals(*(*it2)));
  }
}

TEST_F(SymbolicTest, TestClone) {
  symQc.mcu(xMonom, yMonom, zMonom, {1, 2_nc}, 0);
  const auto clonedQc = symQc;

  symQc.u(xMonom, yMonom, zMonom, 0);
  EXPECT_NE(symQc.getNops(), clonedQc.getNops());
}

TEST_F(SymbolicTest, TestU3SymLambdaPhase) {
  symQc.u(0.0, 0.0, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::P);
}

TEST_F(SymbolicTest, TestU3SymLambdaU2) {
  symQc.u(PI_2, 1.234, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymLambdaU3) {
  symQc.u(4.567, 1.234, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U);
}

TEST_F(SymbolicTest, TestU3SymPhiU2) {
  symQc.u(PI_2, xMonom, 1.234, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymPhiU3) {
  symQc.u(4.567, xMonom, 0.0, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U);

  symQc.u(1.234, xMonom, PI_2, 0);
  EXPECT_EQ((*(symQc.begin() + 1))->getType(), OpType::U);

  symQc.u(3.465, xMonom, PI, 0);
  EXPECT_EQ((*(symQc.begin() + 2))->getType(), OpType::U);

  symQc.u(1.2345, xMonom, 3.465, 0);
  EXPECT_EQ((*(symQc.begin() + 3))->getType(), OpType::U);
}

TEST_F(SymbolicTest, TestU3SymThetaU3) {
  symQc.u(xMonom, 0.0, 0.0, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U);

  symQc.u(xMonom, PI_2, PI_2, 0);
  EXPECT_EQ((*(symQc.begin() + 1))->getType(), OpType::U);

  symQc.u(xMonom, 0.0, PI, 0);
  EXPECT_EQ((*(symQc.begin() + 2))->getType(), OpType::U);

  symQc.u(xMonom, 4.567, 1.234, 0);
  EXPECT_EQ((*(symQc.begin() + 3))->getType(), OpType::U);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymPhiU2) {
  symQc.u(PI_2, xMonom, yMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymPhiU3) {
  symQc.u(PI_2 - 0.2, xMonom, yMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymThetaU3) {
  symQc.u(xMonom, 0.0, yMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U);
}

TEST_F(SymbolicTest, TestU3SymPhiSymThetaU3) {
  symQc.u(xMonom, yMonom, 1.2345, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U);
}

TEST_F(SymbolicTest, TestU2SymLambda) {
  symQc.u2(0.0, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU2SymPhi) {
  symQc.u2(xMonom, 1.2345, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, failPrintingQASM2) {
  symQc.u(xMonom, yMonom, zMonom, 0);
  EXPECT_THROW(std::ignore = symQc.toQASM(false), std::runtime_error);
}

TEST_F(SymbolicTest, failPrintingQASM3) {
  symQc.u(xMonom, yMonom, zMonom, 0);
  EXPECT_THROW(std::ignore = symQc.toQASM(true), std::runtime_error);
}

TEST_F(SymbolicTest, Constructor) {
  EXPECT_NO_THROW(std::ignore =
                      qc::SymbolicOperation(0, 1, qc::OpType::P, {xMonom}));
  EXPECT_NO_THROW(
      std::ignore = qc::SymbolicOperation(0, {1, 2}, qc::OpType::P, {xMonom}));
}
