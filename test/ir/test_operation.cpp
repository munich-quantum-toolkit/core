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
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/AodOperation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"
#include "qasm3/Importer.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

TEST(StandardOperation, CommutesAtQubit) {
  const qc::StandardOperation op1(0, 1, qc::OpType::RY, std::vector{qc::PI_2});
  const qc::StandardOperation op2(0, 1, qc::OpType::RY, std::vector{-qc::PI_4});
  const qc::StandardOperation op3(0, qc::OpType::RY, std::vector{-qc::PI_4});
  EXPECT_TRUE(op1.commutesAtQubit(op2, 0));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 0));
  EXPECT_FALSE(op1.commutesAtQubit(op3, 0));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 2));
}

TEST(CompoundOperation, CommutesAtQubit) {
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector{qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::OpType::RX,
                                          std::vector{qc::PI_2});
  qc::CompoundOperation op2;
  op2.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector{qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector{qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector{qc::PI_2});
  const qc::StandardOperation op3(0, qc::OpType::RY, std::vector{-qc::PI_4});
  const qc::StandardOperation op4(1, qc::OpType::RY, std::vector{-qc::PI_4});
  EXPECT_TRUE(op1.commutesAtQubit(op3, 0));
  EXPECT_TRUE(op3.commutesAtQubit(op1, 0));
  EXPECT_FALSE(op4.commutesAtQubit(op1, 1));
  EXPECT_TRUE(op4.commutesAtQubit(op2, 1));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 0));
  EXPECT_FALSE(op1.commutesAtQubit(op2, 1));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 2));
}

TEST(StandardOperation, IsInverseOf) {
  const qc::StandardOperation op1(0, qc::OpType::RY, std::vector{qc::PI_2});
  qc::StandardOperation op1Inv = op1;
  op1Inv.invert();
  EXPECT_TRUE(op1.isInverseOf(op1Inv));
  EXPECT_FALSE(op1.isInverseOf(op1));
  const qc::StandardOperation op2(0, qc::OpType::Sdg);
  qc::StandardOperation op2Inv = op2;
  op2Inv.invert();
  EXPECT_TRUE(op2.isInverseOf(op2Inv));
  EXPECT_FALSE(op2.isInverseOf(op2));
  const qc::StandardOperation op3(0, qc::OpType::X);
  EXPECT_FALSE(op3.isInverseOf(op1));
}

TEST(CompoundOperation, GlobalIsInverseOf) {
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::RY, std::vector{qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::RY, std::vector{qc::PI_2});
  // the actual inverse of op1
  qc::CompoundOperation op2;
  op2.emplace_back<qc::StandardOperation>(0, qc::RY, std::vector{-qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::RY, std::vector{-qc::PI_2});
  // the compound operations with different number of operations
  qc::CompoundOperation op3 = op2;
  op3.emplace_back<qc::StandardOperation>(2, qc::RY, std::vector{qc::PI_2});
  // the operations come in different order
  qc::CompoundOperation op4;
  op4.emplace_back<qc::StandardOperation>(1, qc::RY, std::vector{-qc::PI_2});
  op4.emplace_back<qc::StandardOperation>(0, qc::RY, std::vector{-qc::PI_2});
  EXPECT_TRUE(op1.isInverseOf(op2));
  EXPECT_TRUE(op2.isInverseOf(op1));
  EXPECT_FALSE(op1.isInverseOf(op3));
  EXPECT_FALSE(op1.isInverseOf(qc::StandardOperation(0, qc::RY)));
  EXPECT_TRUE(op1.isInverseOf(op4));
}

TEST(CompoundOperation, IsInverseOf) {
  // This functionality is not implemented yet, the function isInverseOf leads
  // to false negatives
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector{-qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector{-qc::PI_2});
  qc::CompoundOperation op2 = op1;
  op2.invert();
  EXPECT_TRUE(op1.isInverseOf(op2));
}

TEST(CompoundOperation, GetNqubits) {
  qc::CompoundOperation op;
  op.emplace_back<qc::StandardOperation>(0, qc::OpType::X);
  op.emplace_back<qc::StandardOperation>(1, qc::OpType::X);
  op.emplace_back<qc::StandardOperation>(3, qc::OpType::X);
  EXPECT_EQ(op.getNqubits(), 3);
}

TEST(OpType, General) {
  EXPECT_EQ(qc::toString(qc::RZ), "rz");
  std::stringstream ss;
  ss << qc::OpType::RZ;
  EXPECT_EQ(ss.str(), "rz");
}

TEST(OpType, SingleQubitGate) {
  EXPECT_TRUE(qc::isSingleQubitGate(qc::P));
  EXPECT_FALSE(qc::isSingleQubitGate(qc::ECR));
}

TEST(NonStandardOperation, IsInverseOf) {
  const qc::StandardOperation op(0, qc::I);
  EXPECT_FALSE(qc::NonUnitaryOperation(qc::Targets{0}).isInverseOf(op));
  EXPECT_FALSE(qc::SymbolicOperation().isInverseOf(op));
}

TEST(NonStandardOperation, CommutesAtQubit) {
  const qc::StandardOperation op(0, qc::X);
  EXPECT_FALSE(qc::NonUnitaryOperation(qc::Targets{0}).commutesAtQubit(op, 0));
  EXPECT_FALSE(
      qc::SymbolicOperation(0, qc::P, {sym::Expression<qc::fp, qc::fp>()})
          .commutesAtQubit(op, 0));
}

TEST(Operation, IsIndividualGate) {
  const qc::StandardOperation op1(0, qc::X);
  EXPECT_TRUE(op1.isSingleQubitGate());
  const qc::StandardOperation op2(0, 1, qc::X);
  EXPECT_FALSE(op2.isSingleQubitGate());
  const qc::StandardOperation op3(1, qc::RXX);
  EXPECT_FALSE(op3.isSingleQubitGate());
}

TEST(Operation, IsClifford) {
  const qc::StandardOperation x(0, qc::X);
  EXPECT_TRUE(x.isClifford());
  const qc::StandardOperation cx(0, 1, qc::X);
  EXPECT_TRUE(cx.isClifford());
  const qc::StandardOperation y(0, 1, qc::Y);
  EXPECT_TRUE(y.isClifford());
  const qc::StandardOperation z(0, 1, qc::Z);
  EXPECT_TRUE(z.isClifford());
  const qc::StandardOperation t(0, qc::T);
  EXPECT_FALSE(t.isClifford());
  const qc::Controls controls = {qc::Control(0), qc::Control(1)};
  const qc::StandardOperation ccx(controls, 2, qc::X);
  EXPECT_FALSE(ccx.isClifford());
  const qc::StandardOperation ccy(controls, 2, qc::Y);
  EXPECT_FALSE(ccy.isClifford());
  const qc::StandardOperation ccz(controls, 2, qc::Z);
  EXPECT_FALSE(ccz.isClifford());
  const qc::StandardOperation h(0, qc::H);
  EXPECT_TRUE(h.isClifford());
  const qc::StandardOperation s(0, qc::S);
  EXPECT_TRUE(s.isClifford());
  const qc::StandardOperation sdg(0, qc::Sdg);
  EXPECT_TRUE(sdg.isClifford());
  const qc::StandardOperation sx(0, qc::SX);
  EXPECT_TRUE(sx.isClifford());
  const qc::StandardOperation sxdg(0, qc::SXdg);
  EXPECT_TRUE(sxdg.isClifford());
  const qc::StandardOperation dcx(0, 1, qc::DCX);
  EXPECT_TRUE(dcx.isClifford());
  const qc::StandardOperation swap(0, 1, qc::SWAP);
  EXPECT_TRUE(swap.isClifford());
  const qc::StandardOperation iswap(0, 1, qc::iSWAP);
  EXPECT_TRUE(iswap.isClifford());
  const qc::StandardOperation ecr(0, 1, qc::ECR);
  EXPECT_TRUE(ecr.isClifford());
}

TEST(Operation, IsDiagonalGate) {
  const qc::StandardOperation op1(0, qc::X);
  EXPECT_FALSE(op1.isDiagonalGate());
  const qc::StandardOperation op2(0, qc::Z);
  EXPECT_TRUE(op2.isDiagonalGate());
}

TEST(Operation, IsGlobalGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "rz(pi/4) q[0];\n"
                               "ry(pi/2) q;\n";
  const auto qc = qasm3::Importer::imports(testfile);
  EXPECT_EQ(qc.getHighestLogicalQubitIndex(), 2);
  EXPECT_FALSE(qc.at(0)->isGlobal(3));
  EXPECT_TRUE(qc.at(1)->isGlobal(3));
}

TEST(Operation, Equality) {
  const qc::StandardOperation op1(0, qc::Z);
  const qc::StandardOperation op2(1, 0, qc::Z);
  const qc::StandardOperation op3(0, 1, qc::Z);
  const qc::StandardOperation op4({0, qc::Control::Type::Neg}, 1, qc::Z);
  EXPECT_FALSE(op1 == op2);
  EXPECT_TRUE(op2 == op3);
  EXPECT_TRUE(op3 == op2);
  EXPECT_FALSE(op2 == op4);

  EXPECT_TRUE(op2.equals(op3, qc::Permutation{{{0, 0}, {1, 2}}},
                         qc::Permutation{{{0, 2}, {1, 0}}}));
  EXPECT_FALSE(
      op2.equals(op3, qc::Permutation{{{0, 0}, {1, 2}}}, qc::Permutation{}));
  EXPECT_FALSE(op2.equals(op4, qc::Permutation{{{0, 0}, {1, 2}}},
                          qc::Permutation{{{0, 2}, {1, 0}}}));
}

TEST(StandardOperation, Move) {
  const qc::StandardOperation moveOp({0, 1}, qc::OpType::Move);
  EXPECT_EQ(moveOp.getTargets().size(), 2);
  EXPECT_EQ(moveOp.getNqubits(), 2);
}

TEST(AodOperation, Activate) {
  // activate at position 0, dimension X, start 0.0, end 1.0
  const na::AodOperation activate(qc::OpType::AodActivate, {0},
                                  {na::Dimension::X}, {0.0}, {1.0});
  EXPECT_EQ(activate.getNqubits(), 1);
  EXPECT_EQ(activate.getStarts(na::Dimension::X).at(0), 0.0);
  EXPECT_EQ(activate.getEnds(na::Dimension::X).at(0), 1.0);
}

TEST(AodOperation, Deactivate) {
  // deactivate at position 0,1 dimension Y, start 0.0, end 1.0
  const na::AodOperation deactivate(qc::OpType::AodDeactivate, {0, 1},
                                    {na::Dimension::Y}, {0.0}, {1.0});
  EXPECT_EQ(deactivate.getNqubits(), 2);
  EXPECT_EQ(deactivate.getStarts(na::Dimension::Y).at(0), 0.0);
  EXPECT_EQ(deactivate.getEnds(na::Dimension::Y).at(0), 1.0);
}

TEST(AodOperation, Move) {
  // move from 0,1 to 2,3 dimension X, start 0.0, end 1.0 and dimension Y,
  // start 1.0, end 2.0
  const na::AodOperation move(qc::OpType::AodMove, {0, 1},
                              {na::Dimension::X, na::Dimension::Y}, {0.0, 1.0},
                              {1.0, 2.0});
  EXPECT_EQ(move.getNqubits(), 2);
  EXPECT_EQ(move.getStarts(na::Dimension::X).at(0), 0.0);
  EXPECT_EQ(move.getEnds(na::Dimension::X).at(0), 1.0);
  EXPECT_EQ(move.getStarts(na::Dimension::Y).at(0), 1.0);
  EXPECT_EQ(move.getEnds(na::Dimension::Y).at(0), 2.0);
}

TEST(AodOperation, Distances) {
  const na::AodOperation move(qc::OpType::AodMove, {0, 1},
                              {na::Dimension::X, na::Dimension::Y}, {0.0, 1.0},
                              {1.0, 3.0});
  EXPECT_EQ(move.getMaxDistance(na::Dimension::X), 1.0);
  EXPECT_EQ(move.getMaxDistance(na::Dimension::Y), 2.0);
}

TEST(AodOperation, Qasm) {
  const na::AodOperation move(qc::OpType::AodMove, {0, 1},
                              {na::Dimension::X, na::Dimension::Y}, {0.0, 1.0},
                              {1.0, 3.0});
  std::stringstream ss;
  qc::QuantumRegister qreg(0, 2, "q");
  qc::QubitIndexToRegisterMap qubitToReg{};
  qubitToReg.try_emplace(0, qreg, qreg.toString(0));
  qubitToReg.try_emplace(1, qreg, qreg.toString(1));
  move.dumpOpenQASM(ss, qubitToReg, {}, 0, false);

  EXPECT_EQ(ss.str(), "aod_move (0, 0, 1; 1, 1, 3;) q[0], q[1];\n");
}

TEST(AodOperation, Constructors) {
  uint32_t const dir1 = 0;
  uint32_t const dir2 = 1;

  const na::AodOperation move(qc::OpType::AodMove, {0, 1}, {dir1, dir2},
                              {0.0, 1.0}, {1.0, 3.0});
  const na::AodOperation move2("aod_move", {0, 1}, {dir1}, {0.0}, {1.0});
  const na::AodOperation move3(qc::OpType::AodMove, {0},
                               {std::tuple{na::Dimension::X, 0.0, 1.0}});
  na::SingleOperation const singleOp(na::Dimension::X, 0.0, 1.0);
  const na::AodOperation move4(qc::OpType::AodMove, {0}, {singleOp});

  EXPECT_EQ(0, 0);
}

TEST(AodOperation, OverrideMethods) {
  na::AodOperation move(qc::OpType::AodMove, {0}, {na::Dimension::X}, {0.0},
                        {1.0});
  move.addControl(qc::Control(0, qc::Control::Type::Pos));
  move.removeControl(qc::Control(0, qc::Control::Type::Pos));
  move.clearControls();
  auto it = move.clone();
}

TEST(AodOperation, Invert) {
  na::AodOperation move(qc::OpType::AodMove, {0}, {na::Dimension::X}, {0.0},
                        {1.0});
  move.invert();
  EXPECT_EQ(move.getStarts(na::Dimension::X).at(0), 1.0);
  EXPECT_EQ(move.getEnds(na::Dimension::X).at(0), 0.0);

  na::AodOperation activate(qc::OpType::AodActivate, {0}, {na::Dimension::X},
                            {0.0}, {1.0});
  activate.invert();
  EXPECT_EQ(activate.getType(), qc::OpType::AodDeactivate);

  na::AodOperation deactivate(qc::OpType::AodDeactivate, {0},
                              {na::Dimension::X}, {0.0}, {1.0});
  deactivate.invert();
  EXPECT_EQ(deactivate.getType(), qc::OpType::AodActivate);
}

TEST(StandardOperation, Constructor) {
  EXPECT_NO_THROW(std::ignore =
                      qc::StandardOperation(0, 1, qc::OpType::P, {qc::PI}));
  EXPECT_NO_THROW(
      std::ignore = qc::StandardOperation(0, {1, 2}, qc::OpType::P, {qc::PI}));
}

TEST(StandardOperation, DuplicateQubitThrowsError) {
  EXPECT_THROW(std::ignore =
                   qc::StandardOperation(0, {0, 1}, qc::OpType::P, {qc::PI}),
               std::runtime_error);
  EXPECT_THROW(std::ignore =
                   qc::StandardOperation(0, 0, qc::OpType::P, {qc::PI}),
               std::runtime_error);

  qc::QuantumComputation qc(2);
  EXPECT_THROW(qc.cx(0, 0), std::runtime_error);
  EXPECT_THROW(qc.mcx({0, 1}, 0), std::runtime_error);
}
