/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qasm3/Importer.hpp"
#include "zx/FunctionalityConstruction.hpp"
#include "zx/Simplify.hpp"
#include "zx/ZXDefinitions.hpp"
#include "zx/ZXDiagram.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace zx {
class ZXFunctionalityTest : public ::testing::Test {
public:
  qc::QuantumComputation qc;
};
namespace {
void checkEquivalence(const qc::QuantumComputation& qc1,
                      const qc::QuantumComputation& qc2,
                      const std::vector<qc::Qubit>& qubits) {
  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc1));
  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc2));
  EXPECT_EQ(qc1.getNqubits(), qc2.getNqubits());

  auto d1 = FunctionalityConstruction::buildFunctionality(&qc1);
  auto d2 = FunctionalityConstruction::buildFunctionality(&qc2);
  d1.concat(d2.invert());
  fullReduce(d1);
  EXPECT_TRUE(d1.isIdentity());
  EXPECT_TRUE(d1.globalPhaseIsZero());
  for (const auto q : qubits) {
    ASSERT_LT(q, qc1.getNqubits());
    EXPECT_TRUE(d1.connected(d1.getInput(q), d1.getOutput(q)));
  }
}
} // namespace

TEST_F(ZXFunctionalityTest, parseQasm) {
  const std::string testfile = "OPENQASM 2.0;"
                               "include \"qelib1.inc\";"
                               "qreg q[2];"
                               "h q[0];"
                               "cx q[0],q[1];\n";
  qc = qasm3::Importer::imports(testfile);
  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc));
  const ZXDiagram diag = FunctionalityConstruction::buildFunctionality(&qc);
  EXPECT_EQ(diag.getNVertices(), 7);
  EXPECT_EQ(diag.getNEdges(), 6);

  const auto& inputs = diag.getInputs();
  EXPECT_EQ(inputs[0], 0);
  EXPECT_EQ(inputs[1], 1);

  const auto& outputs = diag.getOutputs();
  EXPECT_EQ(outputs[0], 2);
  EXPECT_EQ(outputs[1], 3);

  constexpr auto edges =
      std::array{std::pair{0U, 4U}, std::pair{5U, 6U}, std::pair{6U, 1U},
                 std::pair{3U, 6U}, std::pair{4U, 5U}, std::pair{5U, 2U}};
  constexpr auto expectedEdgeTypes =
      std::array{EdgeType::Hadamard, EdgeType::Simple, EdgeType::Simple,
                 EdgeType::Simple,   EdgeType::Simple, EdgeType::Simple};
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const auto& [v1, v2] = edges[i];
    const auto& edge = diag.getEdge(v1, v2);
    const auto hasValue = edge.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(edge->type, expectedEdgeTypes[i]);
    }
  }

  constexpr auto expectedVertexTypes = std::array{
      VertexType::Boundary, VertexType::Boundary, VertexType::Boundary,
      VertexType::Boundary, VertexType::Z,        VertexType::Z,
      VertexType::X};
  const auto nVerts = diag.getNVertices();
  for (std::size_t i = 0; i < nVerts; ++i) {
    const auto& vData = diag.getVData(i);
    const auto hasValue = vData.has_value();
    ASSERT_TRUE(hasValue);
    if (hasValue) {
      EXPECT_EQ(vData->type, expectedVertexTypes[i]);
      EXPECT_TRUE(vData->phase.isZero());
    }
  }
}

TEST_F(ZXFunctionalityTest, complexCircuit) {
  std::stringstream ss{};
  ss << "// i 1 0 2\n"
     << "// o 0 1 2\n"
     << "OPENQASM 2.0;" << "include \"qelib1.inc\";" << "qreg q[3];"
     << "sx q[0];" << "sxdg q[0];" << "h q[0];" << "cx q[0],q[1];" << "z q[1];"
     << "x q[2];" << "y q[0];" << "rx(pi/4) q[0];" << "rz(0.1) q[1];"
     << "p(0.1) q[1];" << "ry(pi/4) q[2];" << "t q[0];" << "s q[2];"
     << "u2(pi/4, pi/4) q[1];" << "u3(pi/4, pi/4, pi/4) q[2];"
     << "barrier q[0],q[1],q[2];" << "swap q[0],q[1];" << "cz q[1],q[2];"
     << "cp(pi/4) q[0],q[1];" << "ctrl(2) @ x q[0],q[1],q[2];"
     << "ctrl(2) @ z q[1],q[2],q[0];" << "cp(pi/2) q[0], q[1];"
     << "cp(pi/4) q[0], q[1];" << "cp(pi/8) q[0], q[1];"
     << "rzz(pi/4) q[0], q[1];" << "rxx(pi/4) q[0], q[1];"
     << "ryy(pi/4) q[0], q[1];" << "rzx(pi/4) q[0], q[1];" << "ecr q[0], q[1];"
     << "dcx q[0], q[1];" << "r(pi/8, pi/4) q[2];" << "r(-pi/8, pi/4) q[2];"
     << "dcx q[1], q[0];" << "ecr q[0], q[1];" << "rzx(-pi/4) q[0], q[1];"
     << "ryy(-pi/4) q[0], q[1];" << "rxx(-pi/4) q[0], q[1];"
     << "rzz(-pi/4) q[0], q[1];" << "cp(-pi/8) q[0], q[1];"
     << "cp(-pi/4) q[0], q[1];" << "cp(-pi/2) q[0], q[1];"
     << "ctrl(2) @ z q[1],q[2],q[0];" << "ctrl(2) @ x q[0],q[1],q[2];"
     << "cp(-pi/4) q[0],q[1];" << "cz q[1],q[2];" << "cx q[1],q[0];"
     << "cx q[0],q[1];" << "cx q[1],q[0];" << "u3(-pi/4,-pi/4,-pi/4) q[2];"
     << "u2(-5*pi/4,3*pi/4) q[1];" << "sdg q[2];" << "tdg q[0];"
     << "ry(-pi/4) q[2];" << "p(-0.1) q[1];" << "rz(-0.1) q[1];"
     << "rx(-pi/4) q[0];" << "y q[0];" << "x q[2];" << "z q[1];"
     << "cx q[0],q[1];" << "h q[0];\n";
  qc = qasm3::Importer::import(ss);

  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc));
  ZXDiagram diag = FunctionalityConstruction::buildFunctionality(&qc);
  fullReduce(diag);
  EXPECT_EQ(diag.getNVertices(), 6);
  EXPECT_EQ(diag.getNEdges(), 3);
  EXPECT_TRUE(diag.connected(diag.getInput(0), diag.getOutput(0)));
  EXPECT_TRUE(diag.connected(diag.getInput(1), diag.getOutput(1)));
  EXPECT_TRUE(diag.connected(diag.getInput(2), diag.getOutput(2)));
}

TEST_F(ZXFunctionalityTest, nestedCompoundGate) {
  qc = qc::QuantumComputation(1);
  auto innerOp = std::make_unique<qc::StandardOperation>(0, qc::OpType::X);
  auto compound1 = std::make_unique<qc::CompoundOperation>();
  auto compound2 = std::make_unique<qc::CompoundOperation>();

  compound1->emplace_back(std::move(innerOp));
  compound2->emplace_back(std::move(compound1));

  qc.emplace_back<qc::CompoundOperation>(std::move(compound2));
  qc.x(0);

  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc));
  ZXDiagram diag = FunctionalityConstruction::buildFunctionality(&qc);
  fullReduce(diag);

  EXPECT_TRUE(diag.isIdentity());
}

TEST_F(ZXFunctionalityTest, Phase) {

  qc = qc::QuantumComputation(2);
  qc.p(PI / 4, 0);
  qc.cp(PI / 4, 1, 0);
  qc.cp(-PI / 4, 1, 0);
  qc.p(-PI / 4, 0);

  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc));
  ZXDiagram diag = FunctionalityConstruction::buildFunctionality(&qc);
  fullReduce(diag);

  EXPECT_TRUE(diag.isIdentity());
}

TEST_F(ZXFunctionalityTest, Compound) {
  const std::string testfile =
      "OPENQASM 2.0;"
      "include \"qelib1.inc\";"
      "gate toff q0,q1,q2 {h q2;cx q1,q2;p(-pi/4) q2;cx q0,q2;p(pi/4) q2;cx "
      "q1,q2;p(pi/4) q1;p(-pi/4) q2;cx q0,q2;cx q0,q1;p(pi/4) q0;p(-pi/4) "
      "q1;cx q0,q1;p(pi/4) q2;h q2;}"
      "qreg q[3];"
      "toff q[0],q[1],q[2];"
      "ccx q[0],q[1],q[2];\n";
  qc = qasm3::Importer::imports(testfile);
  EXPECT_TRUE(FunctionalityConstruction::transformableToZX(&qc));
  ZXDiagram diag = FunctionalityConstruction::buildFunctionality(&qc);
  fullReduce(diag);

  EXPECT_TRUE(diag.isIdentity());
}

TEST_F(ZXFunctionalityTest, CRZ) {
  qc = qc::QuantumComputation(2);
  qc.crz(PI / 2, 0, 1);

  auto qcPrime = qc::QuantumComputation(2);

  qcPrime.cx(0, 1);
  qcPrime.rz(-PI / 4, 1);
  qcPrime.cx(0, 1);
  qcPrime.rz(PI / 4, 1);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, CCZ) {

  qc = qc::QuantumComputation(3);
  qc.mcz({1, 2}, 0);

  auto qcPrime = qc::QuantumComputation(3);
  qcPrime.h(0);
  qcPrime.mcx({1, 2}, 0);
  qcPrime.h(0);

  checkEquivalence(qc, qcPrime, {0, 1, 2});
}

TEST_F(ZXFunctionalityTest, MCX) {

  qc = qc::QuantumComputation(4);
  qc.mcx({1, 2, 3}, 0);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcx({3, 2, 1}, 0);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCX0) {

  qc = qc::QuantumComputation(1);
  qc.mcx({}, 0);

  auto qcPrime = qc::QuantumComputation(1);

  qcPrime.x(0);
  checkEquivalence(qc, qcPrime, {0});
}

TEST_F(ZXFunctionalityTest, MCX1) {

  qc = qc::QuantumComputation(2);
  qc.mcx({1}, 0);

  auto qcPrime = qc::QuantumComputation(2);

  qcPrime.cx(1, 0);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, MCZ) {

  qc = qc::QuantumComputation(4);
  qc.mcz({1, 2, 3}, 0);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcz({1, 2, 3}, 0);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCZ0) {

  qc = qc::QuantumComputation(1);
  qc.mcz({}, 0);

  auto qcPrime = qc::QuantumComputation(1);
  qcPrime.z(0);

  checkEquivalence(qc, qcPrime, {0});
}

TEST_F(ZXFunctionalityTest, MCZ1) {

  qc = qc::QuantumComputation(2);
  qc.mcz({1}, 0);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.cz(1, 0);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, MCZ2) {

  qc = qc::QuantumComputation(4);
  qc.mcz({1, 2}, 0);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.h(0);
  qcPrime.mcx({1, 2}, 0);
  qcPrime.h(0);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCRZ) {

  qc = qc::QuantumComputation(4);
  qc.mcrz(PI / 4, {1, 2, 3}, 0);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.h(0);
  qcPrime.mcrx(PI / 4, {1, 2, 3}, 0);
  qcPrime.h(0);
  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCRZ0) {

  qc = qc::QuantumComputation(1);
  qc.mcrz(PI / 4, {}, 0);

  auto qcPrime = qc::QuantumComputation(1);
  qcPrime.rz(PI / 4, 0);

  checkEquivalence(qc, qcPrime, {0});
}

TEST_F(ZXFunctionalityTest, MCRZ1) {

  qc = qc::QuantumComputation(2);
  qc.mcrz(PI / 4, {1}, 0);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.crz(PI / 4, 1, 0);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, UnsupportedControl) {

  qc = qc::QuantumComputation(2);
  qc.cy(1, 0);
  EXPECT_FALSE(FunctionalityConstruction::transformableToZX(&qc));
  EXPECT_THROW(const ZXDiagram diag =
                   FunctionalityConstruction::buildFunctionality(&qc),
               ZXException);
}

TEST_F(ZXFunctionalityTest, UnsupportedControl2) {

  qc = qc::QuantumComputation(3);
  qc.mcy({1, 2}, 0);
  EXPECT_FALSE(FunctionalityConstruction::transformableToZX(&qc));
  EXPECT_THROW(const ZXDiagram diag =
                   FunctionalityConstruction::buildFunctionality(&qc),
               ZXException);
}

TEST_F(ZXFunctionalityTest, InitialLayout) {
  qc = qc::QuantumComputation(2);
  qc::Permutation layout{};
  layout[0] = 1;
  layout[1] = 0;
  qc.initialLayout = layout;
  qc.x(0);
  qc.z(1);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.x(1);
  qcPrime.z(0);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, FromSymbolic) {
  const sym::Variable x{"x"};
  const sym::Term<double> xTerm{x, 1.0};
  qc = qc::QuantumComputation{1};
  qc.rz(qc::Symbolic(xTerm), 0);
  qc.rz(-qc::Symbolic(xTerm), 0);

  ZXDiagram diag = FunctionalityConstruction::buildFunctionality(&qc);

  fullReduce(diag);
  EXPECT_TRUE(diag.isIdentity());
}

TEST_F(ZXFunctionalityTest, RZ) {
  qc = qc::QuantumComputation(1);
  qc.rz(PI / 8, 0);

  auto qcPrime = qc::QuantumComputation(1);
  qcPrime.rz(PI / 8, 0);

  checkEquivalence(qc, qcPrime, {0});
}

TEST_F(ZXFunctionalityTest, ISWAP) {

  qc = qc::QuantumComputation(2);
  qc.iswap(0, 1);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.s(0);
  qcPrime.s(1);
  qcPrime.h(0);
  qcPrime.cx(0, 1);
  qcPrime.cx(1, 0);
  qc.h(1);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, XXplusYY) {
  constexpr auto theta = PI / 4.;
  constexpr auto beta = PI / 2.;

  qc = qc::QuantumComputation(2);
  qc.xx_plus_yy(theta, beta, 0, 1);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.rz(beta, 1);
  qcPrime.rz(-qc::PI_2, 0);
  qcPrime.sx(0);
  qcPrime.rz(qc::PI_2, 0);
  qcPrime.s(1);
  qcPrime.cx(0, 1);
  qcPrime.ry(theta / 2, 0);
  qcPrime.ry(theta / 2, 1);
  qcPrime.cx(0, 1);
  qcPrime.rz(-qc::PI_2, 0);
  qcPrime.sdg(1);
  qcPrime.sxdg(0);
  qcPrime.rz(qc::PI_2, 0);
  qcPrime.rz(-beta, 1);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, XXminusYY) {
  constexpr auto theta = PI / 4.;
  constexpr auto beta = -PI / 2.;

  qc = qc::QuantumComputation(2);
  qc.xx_minus_yy(theta, beta, 0, 1);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.rz(-beta, 1);
  qcPrime.rz(-qc::PI_2, 0);
  qcPrime.sx(0);
  qcPrime.rz(qc::PI_2, 0);
  qcPrime.s(1);
  qcPrime.cx(0, 1);
  qcPrime.ry(-theta / 2, 0);
  qcPrime.ry(theta / 2, 1);
  qcPrime.cx(0, 1);
  qcPrime.sdg(1);
  qcPrime.rz(-qc::PI_2, 0);
  qcPrime.sxdg(0);
  qcPrime.rz(qc::PI_2, 0);
  qcPrime.rz(beta, 1);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, SWAP) {
  qc = qc::QuantumComputation(2);
  qc.swap(0, 1);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.cx(1, 0);
  qcPrime.cx(0, 1);
  qcPrime.cx(1, 0);

  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, CSWAP) {
  qc = qc::QuantumComputation(3);
  qc.mcswap({0}, 1, 2);

  auto qcPrime = qc::QuantumComputation(3);
  qcPrime.mcx({0, 1}, 2);
  qcPrime.mcx({0, 2}, 1);
  qcPrime.mcx({0, 1}, 2);

  checkEquivalence(qc, qcPrime, {0, 1, 2});
}

TEST_F(ZXFunctionalityTest, MCSWAP) {
  qc = qc::QuantumComputation(4);
  qc.mcswap({0, 1}, 2, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcx({0, 1, 2}, 3);
  qcPrime.mcx({0, 1, 3}, 2);
  qcPrime.mcx({0, 1, 2}, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCRzz) {
  qc = qc::QuantumComputation(4);
  qc.mcrzz(qc::PI_2 / 3, {0, 1}, 2, 3);
  qc.mcrzz(2 * qc::PI, {0, 1}, 2, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcrzz(qc::PI_2 / 3, {0, 1}, 2, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCRxx) {
  qc = qc::QuantumComputation(4);
  qc.mcrxx(qc::PI_2, {0, 1}, 2, 3);
  qc.mcrzx(2 * qc::PI, {0, 1}, 2, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcrxx(qc::PI_2, {0, 1}, 2, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCRzx) {
  qc = qc::QuantumComputation(4);
  qc.mcrzx(qc::PI_2, {0, 1}, 2, 3);
  qc.mcrzx(2 * qc::PI, {0, 1}, 2, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcrzx(qc::PI_2, {0, 1}, 2, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCRx0) {
  qc = qc::QuantumComputation(1);
  qc.mcrx(PI / 4, {}, 0);

  auto qcPrime = qc::QuantumComputation(1);
  qcPrime.mcrx(PI / 4, {}, 0);
  checkEquivalence(qc, qcPrime, {0});
}

TEST_F(ZXFunctionalityTest, MCRx1) {
  qc = qc::QuantumComputation(2);
  qc.mcrx(PI / 4, {0}, 1);

  auto qcPrime = qc::QuantumComputation(2);
  qcPrime.mcrx(PI / 4, {0}, 1);
  checkEquivalence(qc, qcPrime, {0, 1});
}

TEST_F(ZXFunctionalityTest, MCRx2) {
  qc = qc::QuantumComputation(3);
  qc.mcrx(PI / 4, {0, 1}, 2);

  auto qcPrime = qc::QuantumComputation(3);
  qcPrime.mcrx(PI / 4, {0, 1}, 2);
  checkEquivalence(qc, qcPrime, {0, 1, 2});
}

TEST_F(ZXFunctionalityTest, MCRx3) {
  qc = qc::QuantumComputation(4);
  qc.mcrx(PI / 4, {0, 1, 2}, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcrx(PI / 4, {0, 1, 2}, 3);
  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCPhase) {
  qc = qc::QuantumComputation(4);
  qc.mcp(PI / 2, {0, 1, 2}, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcp(PI / 2, {0, 1, 2}, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCS) {
  qc = qc::QuantumComputation(4);
  qc.mcs({0, 1, 2}, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mcs({0, 1, 2}, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCT) {
  qc = qc::QuantumComputation(4);
  qc.mct({0, 1, 2}, 3);

  auto qcPrime = qc::QuantumComputation(4);
  qcPrime.mct({0, 1, 2}, 3);

  checkEquivalence(qc, qcPrime, {0, 1, 2, 3});
}

TEST_F(ZXFunctionalityTest, MCPhase2) {
  qc = qc::QuantumComputation(3);
  qc.mcp(PI / 2, {0, 1}, 2);

  auto qcPrime = qc::QuantumComputation(3);
  qcPrime.mcp(PI / 2, {0, 1}, 2);

  checkEquivalence(qc, qcPrime, {0, 1, 2});
}

TEST_F(ZXFunctionalityTest, MCS2) {
  qc = qc::QuantumComputation(3);
  qc.mcs({0, 1}, 2);

  auto qcPrime = qc::QuantumComputation(3);
  qcPrime.mcs({0, 1}, 2);

  checkEquivalence(qc, qcPrime, {0, 1, 2});
}

TEST_F(ZXFunctionalityTest, MCT2) {
  qc = qc::QuantumComputation(3);
  qc.mct({0, 1}, 2);

  auto qcPrime = qc::QuantumComputation(3);
  qcPrime.mct({0, 1}, 2);

  checkEquivalence(qc, qcPrime, {0, 1, 2});
}
} // namespace zx
