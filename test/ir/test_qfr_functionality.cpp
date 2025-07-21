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
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"
#include "qasm3/Importer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qc {

namespace {
void printRegisters(const QuantumComputation& qc) {
  for (const auto& [name, reg] : qc.getQuantumRegisters()) {
    std::cout << "QuantumRegister(name=" << name
              << ", start=" << reg.getStartIndex() << ", size=" << reg.getSize()
              << ")\n";
  }
  for (const auto& [name, reg] : qc.getClassicalRegisters()) {
    std::cout << "ClassicalRegister(name=" << name
              << ", start=" << reg.getStartIndex() << ", size=" << reg.getSize()
              << ")\n";
  }
  for (const auto& [name, reg] : qc.getAncillaRegisters()) {
    std::cout << "AncillaRegister(name=" << name
              << ", start=" << reg.getStartIndex() << ", size=" << reg.getSize()
              << ")\n";
  }
}
} // namespace

class QFRFunctionality : public testing::TestWithParam<std::size_t> {
protected:
  void SetUp() override {
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::ranges::generate(randomData, [&]() { return rd(); });
    std::seed_seq seeds(begin(randomData), end(randomData));
    mt.seed(seeds);
    dist = std::uniform_real_distribution<fp>(0.0, 2 * PI);
  }

  std::mt19937_64 mt;
  std::uniform_real_distribution<fp> dist;
};

TEST_F(QFRFunctionality, removeTrailingIdleQubits) {
  const std::size_t nqubits = 4;
  QuantumComputation qc(nqubits, nqubits);
  qc.x(0);
  qc.x(2);
  std::cout << qc;
  QuantumComputation::printPermutation(qc.outputPermutation);
  printRegisters(qc);

  qc.outputPermutation.erase(1);
  qc.outputPermutation.erase(3);

  qc.stripIdleQubits();
  EXPECT_EQ(qc.getNqubits(), 2);
  std::cout << qc;
  QuantumComputation::printPermutation(qc.outputPermutation);
  printRegisters(qc);

  qc.pop_back();
  qc.outputPermutation.erase(2);
  std::cout << qc;
  QuantumComputation::printPermutation(qc.outputPermutation);
  printRegisters(qc);

  qc.stripIdleQubits();
  EXPECT_EQ(qc.getNqubits(), 1);
}

TEST_F(QFRFunctionality, ancillaryQubitAtEnd) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.addAncillaryRegister(1);
  EXPECT_EQ(qc.getNancillae(), 1);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), 3);
  qc.x(2);
  printRegisters(qc);
  auto p = qc.removeQubit(2);
  EXPECT_EQ(p.first, nqubits);
  EXPECT_EQ(p.second, nqubits);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), nqubits);
  EXPECT_TRUE(qc.getAncillaRegisters().empty());
  printRegisters(qc);
  qc.addAncillaryQubit(p.first, p.second);
  EXPECT_EQ(qc.getNancillae(), 1);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), nqubits + 1);
  EXPECT_FALSE(qc.getAncillaRegisters().empty());
  printRegisters(qc);
  auto q = qc.removeQubit(2);
  EXPECT_EQ(q.first, nqubits);
  EXPECT_EQ(q.second, nqubits);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), nqubits);
  EXPECT_TRUE(qc.getAncillaRegisters().empty());
  printRegisters(qc);
  auto rm = qc.removeQubit(1);
  EXPECT_EQ(rm.first, 1);
  EXPECT_EQ(rm.second, 1);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 1);
  EXPECT_EQ(qc.getNqubits(), 1);
  printRegisters(qc);
  auto empty = qc.removeQubit(0);
  EXPECT_EQ(empty.first, 0);
  EXPECT_EQ(empty.second, 0);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 0);
  EXPECT_EQ(qc.getNqubits(), 0);
  EXPECT_TRUE(qc.getQuantumRegisters().empty());
  printRegisters(qc);
  qc.printStatistics(std::cout);
}

TEST_F(QFRFunctionality, ancillaryQubitRemoveMiddle) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.addAncillaryRegister(3);
  auto p = qc.removeQubit(3);
  EXPECT_EQ(p.first, 3);
  EXPECT_EQ(p.second, 3);
  EXPECT_EQ(qc.getNancillae(), 2);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
  EXPECT_EQ(qc.getNqubits(), 4);
  printRegisters(qc);
}

TEST_F(QFRFunctionality, splitQreg) {
  const std::size_t nqubits = 3;
  QuantumComputation qc(nqubits);
  qc.x(0);
  auto p = qc.removeQubit(1);
  EXPECT_EQ(p.first, 1);
  EXPECT_EQ(p.second, 1);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
  EXPECT_EQ(qc.getNqubits(), 2);
  printRegisters(qc);
}

TEST_F(QFRFunctionality, StripIdleAndDump) {
  const std::string testfile = "OPENQASM 2.0;\n"
                               "include \"qelib1.inc\";\n"
                               "qreg q[5];\n"
                               "creg c[3];\n"
                               "x q[0];\n"
                               "x q[2];\n"
                               "barrier q;\n"
                               "barrier q[0];\n"
                               "reset q;\n"
                               "reset q[2];\n"
                               "cx q[0],q[4];\n";

  auto qc = qasm3::Importer::imports(testfile);
  qc.print(std::cout);
  qc.stripIdleQubits();
  qc.print(std::cout);
  std::stringstream goal{};
  qc.print(goal);
  std::stringstream test{};
  qc.dumpOpenQASM(test, false);
  std::cout << test.str() << "\n";
  qc.reset();
  qc = qasm3::Importer::import(test);
  qc.print(std::cout);
  qc.stripIdleQubits();
  qc.print(std::cout);
  std::stringstream actual{};
  qc.print(actual);
  EXPECT_EQ(goal.str(), actual.str());
}

TEST_F(QFRFunctionality, gateShortCutsAndCloning) {
  // This test checks if the gate shortcuts are working correctly
  // and if the cloning of gates is working correctly.
  // To this end, we create a circuit with every possible gate in the following
  // variants:
  //  - without controls,
  //  - with a single control,
  //  - with multiple controls.
  // Then, we clone the circuit and check if the resulting circuit contains the
  // same number of gates.
  QuantumComputation qc(5, 5);
  qc.gphase(PI);
  qc.i(0);
  qc.ci(1, 0);
  qc.mci({1, 2_nc}, 0);
  qc.h(0);
  qc.ch(1, 0);
  qc.mch({1, 2_nc}, 0);
  qc.x(0);
  qc.cx(1, 0);
  qc.mcx({1, 2_nc}, 0);
  qc.y(0);
  qc.cy(1, 0);
  qc.mcy({1, 2_nc}, 0);
  qc.z(0);
  qc.cz(1, 0);
  qc.mcz({1, 2_nc}, 0);
  qc.s(0);
  qc.cs(1, 0);
  qc.mcs({1, 2_nc}, 0);
  qc.sdg(0);
  qc.csdg(1, 0);
  qc.mcsdg({1, 2_nc}, 0);
  qc.t(0);
  qc.ct(1, 0);
  qc.mct({1, 2_nc}, 0);
  qc.tdg(0);
  qc.ctdg(1, 0);
  qc.mctdg({1, 2_nc}, 0);
  qc.v(0);
  qc.cv(1, 0);
  qc.mcv({1, 2_nc}, 0);
  qc.vdg(0);
  qc.cvdg(1, 0);
  qc.mcvdg({1, 2_nc}, 0);
  qc.u(PI, PI, PI, 0);
  qc.cu(PI, PI, PI, 1, 0);
  qc.mcu(PI, PI, PI, {1, 2_nc}, 0);
  qc.u2(PI, PI, 0);
  qc.cu2(PI, PI, 1, 0);
  qc.mcu2(PI, PI, {1, 2_nc}, 0);
  qc.p(PI, 0);
  qc.cp(PI, 1, 0);
  qc.mcp(PI, {1, 2_nc}, 0);
  qc.sx(0);
  qc.csx(1, 0);
  qc.mcsx({1, 2_nc}, 0);
  qc.sxdg(0);
  qc.csxdg(1, 0);
  qc.mcsxdg({1, 2_nc}, 0);
  qc.rx(PI, 0);
  qc.crx(PI, 1, 0);
  qc.mcrx(PI, {1, 2_nc}, 0);
  qc.ry(PI, 0);
  qc.cry(PI, 1, 0);
  qc.mcry(PI, {1, 2_nc}, 0);
  qc.rz(PI, 0);
  qc.crz(PI, 1, 0);
  qc.mcrz(PI, {1, 2_nc}, 0);
  qc.swap(0, 1);
  qc.cswap(2, 0, 1);
  qc.mcswap({2, 3_nc}, 0, 1);
  qc.iswap(0, 1);
  qc.ciswap(2, 0, 1);
  qc.mciswap({2, 3_nc}, 0, 1);
  qc.iswapdg(0, 1);
  qc.ciswapdg(2, 0, 1);
  qc.mciswapdg({2, 3_nc}, 0, 1);
  qc.peres(0, 1);
  qc.cperes(2, 0, 1);
  qc.mcperes({2, 3_nc}, 0, 1);
  qc.peresdg(0, 1);
  qc.cperesdg(2, 0, 1);
  qc.mcperesdg({2, 3_nc}, 0, 1);
  qc.dcx(0, 1);
  qc.cdcx(2, 0, 1);
  qc.mcdcx({2, 3_nc}, 0, 1);
  qc.ecr(0, 1);
  qc.cecr(2, 0, 1);
  qc.mcecr({2, 3_nc}, 0, 1);
  qc.rxx(PI, 0, 1);
  qc.crxx(PI, 2, 0, 1);
  qc.mcrxx(PI, {2, 3_nc}, 0, 1);
  qc.ryy(PI, 0, 1);
  qc.cryy(PI, 2, 0, 1);
  qc.mcryy(PI, {2, 3_nc}, 0, 1);
  qc.rzz(PI, 0, 1);
  qc.crzz(PI, 2, 0, 1);
  qc.mcrzz(PI, {2, 3_nc}, 0, 1);
  qc.rzx(PI, 0, 1);
  qc.crzx(PI, 2, 0, 1);
  qc.mcrzx(PI, {2, 3_nc}, 0, 1);
  qc.xx_minus_yy(PI, PI, 0, 1);
  qc.cxx_minus_yy(PI, PI, 2, 0, 1);
  qc.mcxx_minus_yy(PI, PI, {2, 3_nc}, 0, 1);
  qc.xx_plus_yy(PI, PI, 0, 1);
  qc.cxx_plus_yy(PI, PI, 2, 0, 1);
  qc.mcxx_plus_yy(PI, PI, {2, 3_nc}, 0, 1);
  qc.measure(0, 0);
  qc.measure({1, 2}, {1, 2});
  qc.barrier(0);
  qc.barrier({1, 2});
  qc.reset(0);
  qc.reset({1, 2});

  const auto qcCloned = qc;
  ASSERT_EQ(qc.size(), qcCloned.size());
  ASSERT_EQ(qcCloned.getGlobalPhase(), PI);
}

TEST_F(QFRFunctionality, cloningDifferentOperations) {
  const std::size_t nqubits = 5;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits);
  comp.barrier(0);
  comp.h(0);
  qc.emplace_back(comp.asOperation());
  qc.classicControlled(X, 0, qc.getClassicalRegisters().at("c"), 1);

  const auto qcCloned = qc;
  ASSERT_EQ(qc.size(), qcCloned.size());
}

TEST_F(QFRFunctionality, wrongRegisterSizes) {
  const std::size_t nqubits = 5;
  QuantumComputation qc(nqubits);
  ASSERT_THROW(qc.measure({0}, {1, 2}), std::runtime_error);
}

TEST_F(QFRFunctionality, OperationEquality) {
  const auto x = StandardOperation(0, X);
  const auto z = StandardOperation(0, Z);
  EXPECT_TRUE(x.equals(x));
  EXPECT_EQ(x, x);
  EXPECT_FALSE(x.equals(z));
  EXPECT_NE(x, z);

  const auto x0 = StandardOperation(0, X);
  const auto x1 = StandardOperation(1, X);
  EXPECT_FALSE(x0.equals(x1));
  EXPECT_NE(x0, x1);
  Permutation perm0{};
  perm0[0] = 1;
  perm0[1] = 0;
  EXPECT_TRUE(x0.equals(x1, perm0, {}));
  EXPECT_TRUE(x0.equals(x1, {}, perm0));

  const auto cx01 = StandardOperation(0, 1, X);
  const auto cx10 = StandardOperation(1, 0, X);
  EXPECT_FALSE(cx01.equals(cx10));
  EXPECT_NE(cx01, cx10);
  EXPECT_FALSE(x0.equals(cx01));
  EXPECT_NE(x0, cx01);

  const auto p = StandardOperation(0, P, {2.0});
  const auto pm = StandardOperation(0, P, {-2.0});
  EXPECT_FALSE(p.equals(pm));
  EXPECT_NE(p, pm);

  const auto measure0 = NonUnitaryOperation(0, 0U);
  const auto measure1 = NonUnitaryOperation(0, 1U);
  const auto measure2 = NonUnitaryOperation(1, 0U);
  EXPECT_FALSE(measure0.equals(x0));
  EXPECT_NE(measure0, x0);
  EXPECT_TRUE(measure0.equals(measure0));
  EXPECT_EQ(measure0, measure0);
  EXPECT_FALSE(measure0.equals(measure1));
  EXPECT_NE(measure0, measure1);
  EXPECT_FALSE(measure0.equals(measure2));
  EXPECT_NE(measure0, measure2);
  EXPECT_TRUE(measure0.equals(measure2, perm0, {}));
  EXPECT_TRUE(measure0.equals(measure2, {}, perm0));

  const auto expectedValue0 = 0U;
  const auto expectedValue1 = 1U;

  std::unique_ptr<Operation> xp0 = std::make_unique<StandardOperation>(0, X);
  std::unique_ptr<Operation> xp1 = std::make_unique<StandardOperation>(0, X);
  std::unique_ptr<Operation> xp2 = std::make_unique<StandardOperation>(0, X);
  const auto classic0 =
      ClassicControlledOperation(std::move(xp0), 0, expectedValue0);
  const auto classic1 =
      ClassicControlledOperation(std::move(xp1), 0, expectedValue1);
  const auto classic2 =
      ClassicControlledOperation(std::move(xp2), 1, expectedValue0);
  std::unique_ptr<Operation> zp = std::make_unique<StandardOperation>(0, Z);
  const auto classic3 =
      ClassicControlledOperation(std::move(zp), 0, expectedValue0);
  EXPECT_FALSE(classic0.equals(x));
  EXPECT_NE(classic0, x);
  EXPECT_TRUE(classic0.equals(classic0));
  EXPECT_EQ(classic0, classic0);
  EXPECT_FALSE(classic0.equals(classic1));
  EXPECT_NE(classic0, classic1);
  EXPECT_FALSE(classic0.equals(classic2));
  EXPECT_NE(classic0, classic2);
  EXPECT_FALSE(classic0.equals(classic3));
  EXPECT_NE(classic0, classic3);

  auto compound0 = CompoundOperation();
  compound0.emplace_back<StandardOperation>(0, X);

  auto compound1 = CompoundOperation();
  compound1.emplace_back<StandardOperation>(0, X);
  compound1.emplace_back<StandardOperation>(0, Z);

  auto compound2 = CompoundOperation();
  compound2.emplace_back<StandardOperation>(0, Z);

  EXPECT_FALSE(compound0.equals(x));
  EXPECT_NE(compound0, x);
  EXPECT_TRUE(compound0.equals(compound0));
  EXPECT_EQ(compound0, compound0);
  EXPECT_FALSE(compound0.equals(compound1));
  EXPECT_NE(compound0, compound1);
  EXPECT_FALSE(compound0.equals(compound2));
  EXPECT_NE(compound0, compound2);
}

TEST_F(QFRFunctionality, IndexOutOfRange) {
  QuantumComputation qc(2);
  Permutation layout{};
  layout[0] = 0;
  layout[2] = 1;
  qc.initialLayout = layout;
  qc.x(0);

  EXPECT_THROW(qc.x(1), std::out_of_range);
  EXPECT_THROW(qc.cx(1_nc, 0), std::out_of_range);
  EXPECT_THROW(qc.mcx({2_nc, 1_nc}, 0), std::out_of_range);
  EXPECT_THROW(qc.swap(0, 1), std::out_of_range);
  EXPECT_THROW(qc.cswap(1_nc, 0, 2), std::out_of_range);
  EXPECT_THROW(qc.reset({0, 1, 2}), std::out_of_range);
}

TEST_F(QFRFunctionality, ContainsLogicalQubit) {
  const QuantumComputation qc(2);
  const auto [contains0, index0] = qc.containsLogicalQubit(0);
  EXPECT_TRUE(contains0);
  const auto hasValue0 = index0.has_value();
  ASSERT_TRUE(hasValue0);
  if (hasValue0) {
    EXPECT_EQ(*index0, 0);
  }
  const auto [contains1, index1] = qc.containsLogicalQubit(1);
  EXPECT_TRUE(contains1);
  const auto hasValue1 = index1.has_value();
  ASSERT_TRUE(hasValue1);
  if (hasValue1) {
    EXPECT_EQ(*index1, 1);
  }
  const auto [contains2, index2] = qc.containsLogicalQubit(2);
  EXPECT_FALSE(contains2);
  EXPECT_FALSE(index2.has_value());
}

TEST_F(QFRFunctionality, AddAncillaryQubits) {
  QuantumComputation qc(1);
  qc.addAncillaryQubit(1, std::nullopt);
  EXPECT_EQ(qc.getNqubits(), 2);
  EXPECT_EQ(qc.getNancillae(), 1);
  ASSERT_EQ(qc.getAncillary().size(), 2U);
  ASSERT_EQ(qc.getGarbage().size(), 2U);
  EXPECT_FALSE(qc.getAncillary()[0]);
  EXPECT_TRUE(qc.getAncillary()[1]);
  EXPECT_FALSE(qc.getGarbage()[0]);
  EXPECT_TRUE(qc.getGarbage()[1]);
}

TEST_F(QFRFunctionality, CircuitDepthEmptyCircuit) {
  const QuantumComputation qc(2);
  EXPECT_EQ(qc.getDepth(), 0U);
}

TEST_F(QFRFunctionality, CircuitDepthStandardOperations) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.h(1);
  qc.cx(1, 0);

  EXPECT_EQ(qc.getDepth(), 2U);
}

TEST_F(QFRFunctionality, CircuitDepthNonUnitaryOperations) {
  QuantumComputation qc(2U, 2U);
  qc.h(0);
  qc.h(1);
  qc.cx(1, 0);
  qc.barrier({0, 1});
  qc.measure(0, 0);
  qc.measure(1, 1);

  EXPECT_EQ(qc.getDepth(), 3U);
}

// Test with compound operation
TEST_F(QFRFunctionality, CircuitDepthCompoundOperation) {
  QuantumComputation comp(2);
  comp.h(0);
  comp.h(1);
  comp.cx(1, 0);

  QuantumComputation qc(2);
  qc.emplace_back(comp.asOperation());

  EXPECT_EQ(qc.getDepth(), 2U);
}

TEST_F(QFRFunctionality, CircuitToOperation) {
  QuantumComputation qc(2U, 2U);
  EXPECT_EQ(qc.asOperation(), nullptr);
  qc.x(0);
  const auto& op = qc.asOperation();
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->getType(), X);
  EXPECT_EQ(op->getNcontrols(), 0U);
  EXPECT_EQ(op->getTargets().front(), 0U);
  EXPECT_TRUE(qc.empty());
  qc.x(0);
  qc.h(0);
  qc.classicControlled(X, 0, 1, {0, 1U}, 1U);
  const auto& op2 = qc.asOperation();
  ASSERT_NE(op2, nullptr);
  EXPECT_EQ(op2->getType(), Compound);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, AvoidStrippingIdleQubitWhenInOutputPermutation) {
  // a qubit being present in the output permutation should not be stripped
  QuantumComputation qc(2U, 2U);
  qc.measure(1, 0);
  qc.initializeIOMapping();
  qc.stripIdleQubits();
  EXPECT_EQ(qc.getNqubits(), 2U);
  EXPECT_EQ(qc.outputPermutation[1], 0U);
}

TEST_F(QFRFunctionality, UpdateOutputPermutation) {
  // Update output permutation if swap gate was applied even if physical qubit
  // index matches logical qubit index
  QuantumComputation qc(5U, 3U);
  qc.h(0);

  // Swap qubits 2 and 3
  qc.swap(2, 3);

  qc.cx(1, 0);
  qc.cx(1, 3);
  qc.initialLayout[1] = 0;
  qc.initialLayout[0] = 1;
  qc.initialLayout[3] = 2;
  qc.initialLayout[2] = 3;
  qc.initialLayout[4] = 4;
  qc.measure(1, 0);
  qc.measure(0, 1);
  qc.measure(2, 2);
  qc.initializeIOMapping();

  // Check that output permutation is equal to initialLayout for the measured
  // qubits, except for the swap between qubits 2 and 3
  EXPECT_EQ(qc.outputPermutation[1], qc.initialLayout[1]);
  EXPECT_EQ(qc.outputPermutation[0], qc.initialLayout[0]);
  EXPECT_EQ(qc.outputPermutation[2], qc.initialLayout[3]);
}

TEST_F(QFRFunctionality, RzAndPhaseDifference) {
  const std::string qasm = "// i 0 1\n"
                           "// o 0 1\n"
                           "OPENQASM 2.0;\n"
                           "include \"qelib1.inc\";\n"
                           "qreg q[2];\n"
                           "rz(1/8) q[0];\n"
                           "p(1/8) q[1];\n"
                           "crz(1/8) q[0],q[1];\n"
                           "cp(1/8) q[0],q[1];\n";
  auto qc = qasm3::Importer::imports(qasm);
  std::cout << qc << "\n";
  std::stringstream oss;
  qc.dumpOpenQASM(oss, false);
}

TEST_F(QFRFunctionality, U3toU2Gate) {
  QuantumComputation qc(1);
  qc.u(PI_2, 0., PI, 0);      // H
  qc.u(PI_2, 0., 0., 0);      // RY(pi/2)
  qc.u(PI_2, -PI_2, PI_2, 0); // V = RX(pi/2)
  qc.u(PI_2, PI_2, -PI_2, 0); // Vdag = RX(-pi/2)
  qc.u(PI_2, 0.25, 0.5, 0);   // U2(0.25, 0.5)
  std::cout << qc << "\n";
  EXPECT_EQ(qc.at(0)->getType(), H);
  EXPECT_EQ(qc.at(1)->getType(), RY);
  EXPECT_EQ(qc.at(1)->getParameter().at(0), PI_2);
  EXPECT_EQ(qc.at(2)->getType(), V);
  EXPECT_EQ(qc.at(3)->getType(), Vdg);
  EXPECT_EQ(qc.at(4)->getType(), U2);
  EXPECT_EQ(qc.at(4)->getParameter().at(0), 0.25);
  EXPECT_EQ(qc.at(4)->getParameter().at(1), 0.5);
}

TEST_F(QFRFunctionality, U3toU1Gate) {
  QuantumComputation qc(1);
  qc.u(0., 0., 0., 0);    // I
  qc.u(0., 0., PI, 0);    // Z
  qc.u(0., 0., PI_2, 0);  // S
  qc.u(0., 0., -PI_2, 0); // Sdg
  qc.u(0., 0., PI_4, 0);  // T
  qc.u(0., 0., -PI_4, 0); // Tdg
  qc.u(0., 0., 0.5, 0);   // p(0.5)

  std::cout << qc << "\n";
  EXPECT_EQ(qc.at(0)->getType(), I);
  EXPECT_EQ(qc.at(1)->getType(), Z);
  EXPECT_EQ(qc.at(2)->getType(), S);
  EXPECT_EQ(qc.at(3)->getType(), Sdg);
  EXPECT_EQ(qc.at(4)->getType(), T);
  EXPECT_EQ(qc.at(5)->getType(), Tdg);
  EXPECT_EQ(qc.at(6)->getType(), P);
  EXPECT_EQ(qc.at(6)->getParameter().at(0), 0.5);
}

TEST_F(QFRFunctionality, U3SpecialCases) {
  QuantumComputation qc(1);
  qc.u(0.5, 0., 0., 0);      // RY(0.5)
  qc.u(0.5, -PI_2, PI_2, 0); // RX(0.5)
  qc.u(0.5, PI_2, -PI_2, 0); // RX(-0.5)
  qc.u(PI, PI_2, PI_2, 0);   // Y
  qc.u(PI, 0., PI, 0);       // X
  qc.u(0.5, 0.25, 0.125, 0); // U3(0.5, 0.25, 0.125)

  std::cout << qc << "\n";
  EXPECT_EQ(qc.at(0)->getType(), RY);
  EXPECT_EQ(qc.at(0)->getParameter().at(0), 0.5);
  EXPECT_EQ(qc.at(1)->getType(), RX);
  EXPECT_EQ(qc.at(1)->getParameter().at(0), 0.5);
  EXPECT_EQ(qc.at(2)->getType(), RX);
  EXPECT_EQ(qc.at(2)->getParameter().at(0), -0.5);
  EXPECT_EQ(qc.at(3)->getType(), Y);
  EXPECT_EQ(qc.at(4)->getType(), X);
  EXPECT_EQ(qc.at(5)->getType(), U);
  EXPECT_EQ(qc.at(5)->getParameter().at(0), 0.5);
  EXPECT_EQ(qc.at(5)->getParameter().at(1), 0.25);
  EXPECT_EQ(qc.at(5)->getParameter().at(2), 0.125);
}

TEST_F(QFRFunctionality, GlobalPhaseNormalization) {
  QuantumComputation qc(1);
  EXPECT_EQ(qc.getGlobalPhase(), 0.);
  qc.gphase(-PI);
  EXPECT_EQ(qc.getGlobalPhase(), PI);
  qc.gphase(PI);
  EXPECT_EQ(qc.getGlobalPhase(), 0.);
}

TEST_F(QFRFunctionality, OpNameToTypeSimple) {
  EXPECT_EQ(OpType::X, opTypeFromString("x"));
  EXPECT_EQ(OpType::Y, opTypeFromString("y"));
  EXPECT_EQ(OpType::Z, opTypeFromString("z"));

  EXPECT_EQ(OpType::H, opTypeFromString("h"));
  EXPECT_EQ(OpType::S, opTypeFromString("s"));
  EXPECT_EQ(OpType::Sdg, opTypeFromString("sdg"));
  EXPECT_EQ(OpType::T, opTypeFromString("t"));
  EXPECT_EQ(OpType::Tdg, opTypeFromString("tdg"));

  EXPECT_EQ(OpType::X, opTypeFromString("cnot"));

  EXPECT_THROW([[maybe_unused]] const auto type = opTypeFromString("foo"),
               std::invalid_argument);
}

TEST_F(QFRFunctionality, addControlStandardOperation) {
  auto op = StandardOperation(0, OpType::X);
  op.addControl(1);
  op.addControl(2);
  ASSERT_EQ(op.getNcontrols(), 2);
  const auto expectedControls = Controls{1U, 2U};
  EXPECT_EQ(op.getControls(), expectedControls);
  op.removeControl(1);
  const auto expectedControlsAfterRemove = Controls{2U};
  EXPECT_EQ(op.getControls(), expectedControlsAfterRemove);
  op.clearControls();
  EXPECT_EQ(op.getNcontrols(), 0);
  ASSERT_THROW(op.removeControl(1), std::runtime_error);

  op.addControl(1);
  const auto& controls = op.getControls();
  EXPECT_EQ(op.removeControl(controls.begin()), controls.end());
}

TEST_F(QFRFunctionality, addControlSymbolicOperation) {
  auto op = SymbolicOperation(0, OpType::X);

  op.addControl(1);
  op.addControl(2);

  ASSERT_EQ(op.getNcontrols(), 2);
  auto expectedControls = Controls{1U, 2U};
  EXPECT_EQ(op.getControls(), expectedControls);
  op.removeControl(1);
  auto expectedControlsAfterRemove = Controls{2U};
  EXPECT_EQ(op.getControls(), expectedControlsAfterRemove);
  op.clearControls();
  EXPECT_EQ(op.getNcontrols(), 0);

  op.addControl(1);
  const auto& controls = op.getControls();
  EXPECT_EQ(op.removeControl(controls.begin()), controls.end());
}

TEST_F(QFRFunctionality, addControlClassicControlledOperation) {
  std::unique_ptr<Operation> xp = std::make_unique<StandardOperation>(0, X);
  const auto expectedValue = 0U;
  auto op = ClassicControlledOperation(std::move(xp), 0, expectedValue);

  op.addControl(1);
  op.addControl(2);

  ASSERT_EQ(op.getNcontrols(), 2);
  auto expectedControls = Controls{1U, 2U};
  EXPECT_EQ(op.getControls(), expectedControls);
  op.removeControl(1);
  auto expectedControlsAfterRemove = Controls{2U};
  EXPECT_EQ(op.getControls(), expectedControlsAfterRemove);
  op.clearControls();
  EXPECT_EQ(op.getNcontrols(), 0);
  op.addControl(1);
  const auto& controls = op.getControls();
  EXPECT_EQ(op.removeControl(controls.begin()), controls.end());
}

TEST_F(QFRFunctionality, addControlNonUnitaryOperation) {
  auto op = NonUnitaryOperation(0U, Measure);

  EXPECT_THROW(op.addControl(1), std::runtime_error);
  EXPECT_THROW(op.removeControl(1), std::runtime_error);
  EXPECT_THROW(op.clearControls(), std::runtime_error);
  // we pass an invalid iterator to removeControl, which is fine, since the
  // function call should unconditionally trap
  EXPECT_THROW(op.removeControl(Controls::const_iterator{}),
               std::runtime_error);
}

TEST_F(QFRFunctionality, addControlCompoundOperation) {
  auto op = CompoundOperation();

  auto control0 = 0U;
  auto control1 = 1U;

  auto xOp = std::make_unique<StandardOperation>(Targets{1}, OpType::X);
  auto cxOp = std::make_unique<StandardOperation>(Targets{3}, OpType::X);
  cxOp->addControl(control1);

  op.emplace_back(xOp);
  op.emplace_back(cxOp);

  op.addControl(control0);

  ASSERT_EQ(op.getOps()[0]->getNcontrols(), 1);
  ASSERT_EQ(op.getOps()[1]->getNcontrols(), 2);

  op.clearControls();

  ASSERT_EQ(op.getOps()[0]->getNcontrols(), 0);
  ASSERT_EQ(op.getOps()[1]->getNcontrols(), 1);
  ASSERT_EQ(*op.getOps()[1]->getControls().begin(), control1);
  EXPECT_THROW(op.removeControl(control0), std::runtime_error);
}

TEST_F(QFRFunctionality, addControlTwice) {
  auto control = 0U;

  std::unique_ptr<Operation> op =
      std::make_unique<StandardOperation>(Targets{1}, OpType::X);
  op->addControl(control);
  EXPECT_THROW(op->addControl(control), std::runtime_error);

  auto classicControlledOp = ClassicControlledOperation(std::move(op), 0, 0U);
  EXPECT_THROW(classicControlledOp.addControl(control), std::runtime_error);

  auto symbolicOp = SymbolicOperation(Targets{1}, OpType::X);
  symbolicOp.addControl(control);
  EXPECT_THROW(symbolicOp.addControl(control), std::runtime_error);
}

TEST_F(QFRFunctionality, addTargetAsControl) {
  // Adding a control that is already a target
  auto control = 1U;

  std::unique_ptr<Operation> op =
      std::make_unique<StandardOperation>(Targets{1}, OpType::X);
  EXPECT_THROW(op->addControl(control), std::runtime_error);

  auto classicControlledOp = ClassicControlledOperation(std::move(op), 0, 0U);
  EXPECT_THROW(classicControlledOp.addControl(control), std::runtime_error);

  auto symbolicOp = SymbolicOperation(Targets{1}, OpType::X);
  EXPECT_THROW(symbolicOp.addControl(control), std::runtime_error);
}

TEST_F(QFRFunctionality, addControlCompoundOperationInvalid) {
  auto op = CompoundOperation();

  auto control1 = 1U;

  auto xOp = std::make_unique<StandardOperation>(Targets{1}, OpType::X);
  auto cxOp = std::make_unique<StandardOperation>(Targets{3}, OpType::X);
  cxOp->addControl(control1);

  op.emplace_back(xOp);
  op.emplace_back(cxOp);

  ASSERT_THROW(op.addControl(control1), std::runtime_error);
  ASSERT_THROW(op.addControl(Control{1}), std::runtime_error);
}

TEST_F(QFRFunctionality, invertUnsupportedOperation) {
  auto op = NonUnitaryOperation(0U, OpType::Measure);

  ASSERT_THROW(op.invert(), std::runtime_error);
}

TEST_F(QFRFunctionality, invertStandardOpSelfInverting) {
  const auto opTypes = {
      OpType::I, OpType::X, OpType::Y, OpType::Z, OpType::H, OpType::SWAP,
  };

  for (auto opType : opTypes) {
    auto op = StandardOperation(0U, opType);
    op.invert();
    ASSERT_EQ(op.getType(), opType);
  }
}

TEST_F(QFRFunctionality, invertStandardOpInvertClone) {
  auto op1 = StandardOperation(0U, S);
  auto op2 = op1.getInverted();
  ASSERT_EQ(op1.getType(), S);
  ASSERT_EQ(op2->getType(), Sdg);
}

TEST_F(QFRFunctionality, invertStandardOpSpecial) {
  const auto opTypes = {
      std::pair{S, Sdg},   std::pair{T, Tdg},         std::pair{V, Vdg},
      std::pair{SX, SXdg}, std::pair{Peres, Peresdg}, std::pair{iSWAP, iSWAPdg},
  };

  for (const auto& [opType, opTypeInv] : opTypes) {
    auto op = StandardOperation(0U, opType);
    op.invert();
    ASSERT_EQ(op.getType(), opTypeInv);

    auto op2 = StandardOperation(0U, opTypeInv);
    op2.invert();
    ASSERT_EQ(op2.getType(), opType);
  }
}

TEST_F(QFRFunctionality, invertStandardOpParamChange) {
  const auto cases = {
      std::tuple{OpType::GPhase, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::P, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RX, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RY, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RZ, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RXX, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RYY, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RZZ, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RZX, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::U2, std::vector<fp>{1, 1},
                 std::vector<fp>{-1 + PI, -1 - PI}},
      std::tuple{OpType::U, std::vector<fp>{1, 2, 3},
                 std::vector<fp>{-1, -3, -2}},
      std::tuple{OpType::XXminusYY, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::XXplusYY, std::vector<fp>{1}, std::vector<fp>{-1}},
  };

  for (const auto& testcase : cases) {
    auto op =
        StandardOperation(0U, std::get<0>(testcase), std::get<1>(testcase));
    op.invert();
    ASSERT_EQ(op.getParameter(), std::get<2>(testcase));
  }

  auto op = StandardOperation(Targets{0U, 1U}, OpType::DCX);
  op.invert();
  const auto expectedTargets = Targets{1U, 0U};
  ASSERT_EQ(op.getTargets(), expectedTargets);
}

TEST_F(QFRFunctionality, invertCompoundOperation) {
  auto op = CompoundOperation();

  op.emplace_back<StandardOperation>(0U, OpType::X);
  op.emplace_back<StandardOperation>(1U, OpType::RZ, std::vector<fp>{1});
  op.emplace_back<StandardOperation>(1U, OpType::S);

  op.invert();

  ASSERT_EQ(op.getOps()[0]->getType(), OpType::Sdg);
  ASSERT_EQ(op.getOps()[1]->getType(), OpType::RZ);
  ASSERT_EQ(op.getOps()[1]->getParameter(), std::vector<fp>{-1});
  ASSERT_EQ(op.getOps()[2]->getType(), OpType::X);
}

TEST_F(QFRFunctionality, invertSymbolicOpParamChange) {
  auto x = sym::Variable("x");
  auto y = sym::Variable("y");
  const auto cases = {
      std::tuple{OpType::GPhase, std::vector<SymbolOrNumber>{Symbolic({x})},
                 std::vector<SymbolOrNumber>{-Symbolic({x})}},
      std::tuple{OpType::GPhase, std::vector<SymbolOrNumber>{1.0},
                 std::vector<SymbolOrNumber>{-1.0}},
      std::tuple{OpType::U2, std::vector<SymbolOrNumber>{Symbolic({x}), 1.0},
                 std::vector<SymbolOrNumber>{-1.0 + PI, -Symbolic({x}) - PI}},
      std::tuple{
          OpType::U,
          std::vector<SymbolOrNumber>{Symbolic({x}), 2.0, Symbolic({y})},
          std::vector<SymbolOrNumber>{-Symbolic({x}), -Symbolic({y}), -2.0}},
      std::tuple{OpType::XXminusYY, std::vector<SymbolOrNumber>{Symbolic({x})},
                 std::vector<SymbolOrNumber>{-Symbolic({x})}},
      std::tuple{OpType::XXplusYY, std::vector<SymbolOrNumber>{1.0},
                 std::vector<SymbolOrNumber>{-1.0}},
  };

  for (const auto& testcase : cases) {
    auto op =
        SymbolicOperation(0U, std::get<0>(testcase), std::get<1>(testcase));
    op.invert();

    for (size_t i = 0; i < std::get<1>(testcase).size(); ++i) {
      ASSERT_EQ(op.getParameter(i), std::get<2>(testcase)[i]);
    }
  }

  // The following gate should be handled by the StandardOperation function
  auto op = SymbolicOperation(Targets{0U, 1U}, OpType::DCX);
  op.invert();
  const auto expectedTargets = Targets{1U, 0U};
  ASSERT_EQ(op.getTargets(), expectedTargets);
}

TEST_F(QFRFunctionality, measureAll) {
  QuantumComputation qc(2U);
  qc.measureAll();
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.getNcbits(), 2U);
  EXPECT_EQ(qc.getClassicalRegisters().size(), 1U);
}

TEST_F(QFRFunctionality, measureAllExistingRegister) {
  QuantumComputation qc(2U, 2U);
  qc.measureAll(false);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.getNcbits(), 2U);
  EXPECT_EQ(qc.getClassicalRegisters().size(), 1U);
}

TEST_F(QFRFunctionality, measureAllInsufficientRegisterSize) {
  QuantumComputation qc(2U, 1U);
  EXPECT_THROW(qc.measureAll(false), std::runtime_error);
}

TEST_F(QFRFunctionality, checkClassicalRegisters) {
  QuantumComputation qc(1U, 1U);
  EXPECT_THROW(qc.classicControlled(X, 0U, {0U, 2U}), std::runtime_error);
}

TEST_F(QFRFunctionality, testSettingAncillariesProperlyCreatesRegisters) {
  // create an empty circuit and assert some properties about its registers
  QuantumComputation qc(3U);
  const auto& qregs = qc.getQuantumRegisters();
  ASSERT_EQ(qregs.size(), 1U);
  const auto& reg = *qregs.begin();
  const auto name = reg.first;
  ASSERT_EQ(reg.second.getStartIndex(), 0U);
  ASSERT_EQ(reg.second.getSize(), 3U);
  const auto& ancRegs = qc.getAncillaRegisters();
  ASSERT_TRUE(ancRegs.empty());
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 3U);
  ASSERT_EQ(qc.getNancillae(), 0U);

  // set some ancillaries and assert that the registers are created properly
  qc.setLogicalQubitAncillary(2U);
  qc.setLogicalQubitAncillary(1U);
  ASSERT_EQ(qregs.size(), 1U);
  ASSERT_EQ(reg.second.getStartIndex(), 0U);
  ASSERT_EQ(reg.second.getSize(), 3U);
  ASSERT_EQ(name, reg.first);
  ASSERT_TRUE(ancRegs.empty());
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 1U);
  ASSERT_EQ(qc.getNancillae(), 2U);

  // add one gate to the circuit, mark the last two qubits as garbage and call
  // the `stripIdleQubits` method to remove the (idle) ancillary qubits. Then,
  // assert that the registers are still correct.
  qc.x(0);
  qc.setLogicalQubitGarbage(1U);
  qc.setLogicalQubitGarbage(2U);
  qc.stripIdleQubits();
  ASSERT_EQ(qregs.size(), 1U);
  ASSERT_EQ(reg.second.getStartIndex(), 0U);
  ASSERT_EQ(reg.second.getSize(), 1U);
  ASSERT_EQ(name, reg.first);
  ASSERT_TRUE(ancRegs.empty());
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 1U);
  ASSERT_EQ(qc.getNancillae(), 0U);
}

TEST_F(QFRFunctionality, testSettingSetMultipleAncillariesAndGarbage) {
  // create an empty circuit and assert some properties about its registers
  QuantumComputation qc(3U);
  const auto& ancRegs = qc.getAncillaRegisters();
  ASSERT_TRUE(ancRegs.empty());
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 3U);
  ASSERT_EQ(qc.getNancillae(), 0U);

  // set some ancillaries garbage and assert that the registers are created
  // properly
  qc.setLogicalQubitsAncillary(1U, 2U);
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 1U);
  ASSERT_EQ(qc.getNancillae(), 2U);
  qc.setLogicalQubitsGarbage(1U, 2U);
  ASSERT_EQ(qc.getNgarbageQubits(), 2U);
  ASSERT_EQ(qc.getNmeasuredQubits(), 1U);
}

TEST_F(QFRFunctionality, StripIdleQubitsInMiddleOfCircuit) {
  QuantumComputation qc{};
  qc.addQubitRegister(5, "q");
  qc.setLogicalQubitAncillary(3U);
  qc.setLogicalQubitAncillary(4U);
  qc.setLogicalQubitGarbage(3U);
  qc.setLogicalQubitGarbage(4U);
  qc.initialLayout.clear();
  qc.initialLayout[0U] = 3U;
  qc.initialLayout[1U] = 0U;
  qc.initialLayout[2U] = 4U;
  qc.initialLayout[3U] = 2U;
  qc.initialLayout[4U] = 1U;
  qc.outputPermutation.clear();
  qc.outputPermutation[1U] = 2U;
  qc.outputPermutation[3U] = 0U;
  qc.outputPermutation[4U] = 1U;

  qc.x(1);
  qc.x(3);
  qc.x(4);

  const auto& qregs = qc.getQuantumRegisters();
  ASSERT_EQ(qregs.size(), 1U);
  const auto& [name, reg] = *qregs.begin();
  ASSERT_EQ(name, "q");
  ASSERT_EQ(reg.getStartIndex(), 0U);
  ASSERT_EQ(reg.getSize(), 5U);
  const auto& ancRegs = qc.getAncillaRegisters();
  ASSERT_TRUE(ancRegs.empty());
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 3U);
  ASSERT_EQ(qc.getNancillae(), 2U);

  qc.stripIdleQubits();

  ASSERT_EQ(qregs.size(), 2U);
  const auto& regAfter = qregs.at("q_l");
  ASSERT_EQ(regAfter.getStartIndex(), 1U);
  ASSERT_EQ(regAfter.getSize(), 1U);
  const auto& reg2After = qregs.at("q_h");
  ASSERT_EQ(reg2After.getStartIndex(), 3U);
  ASSERT_EQ(reg2After.getSize(), 2U);
  ASSERT_TRUE(ancRegs.empty());
  ASSERT_EQ(qc.getNqubitsWithoutAncillae(), 3U);
  ASSERT_EQ(qc.getNancillae(), 0U);
}

TEST_F(QFRFunctionality, trivialOperationReordering) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.h(1);
  std::cout << qc << "\n";
  qc.reorderOperations();
  std::cout << qc << "\n";
  auto it = qc.begin();
  const auto target = (*it)->getTargets().at(0);
  EXPECT_EQ(target, 1);
  ++it;
  const auto target2 = (*it)->getTargets().at(0);
  EXPECT_EQ(target2, 0);
}

TEST_F(QFRFunctionality, OperationReorderingBarrier) {
  QuantumComputation qc(3);
  qc.h(0);
  qc.barrier({0, 1});
  qc.h(1);
  std::cout << qc << "\n";
  qc.reorderOperations();
  std::cout << qc << "\n";
  auto it = qc.begin();
  const auto target = (*it)->getTargets().at(0);
  EXPECT_EQ(target, 0);
  ++it;
  ++it;
  const auto target2 = (*it)->getTargets().at(0);
  EXPECT_EQ(target2, 1);
}

TEST_F(QFRFunctionality, isDynamicCompoundOperation) {
  QuantumComputation qc(1, 1);
  QuantumComputation compound(1, 1);
  compound.measure(0, 0);
  compound.x(0);
  compound.measure(0, 0);
  qc.emplace_back(compound.asCompoundOperation());
  std::cout << qc << "\n";
  EXPECT_TRUE(qc.isDynamic());
}

TEST_F(QFRFunctionality, emptyPermutation) {
  const Permutation perm{};

  EXPECT_EQ(perm.size(), 0U);
  EXPECT_EQ(perm.apply(0U), 0U);
  EXPECT_EQ(perm.apply(Targets{0U}), Targets{0U});
  EXPECT_EQ(perm.apply(Controls{0U}), Controls{0U});
  EXPECT_EQ(perm.maxKey(), 0U);
  EXPECT_EQ(perm.maxValue(), 0U);
}

TEST_F(QFRFunctionality, NoRegisterOnEmptyCircuit) {
  // This is a regression test. Previously, the following code would throw an
  // exception because even zero-qubit circuits had an empty register named "q".
  QuantumComputation qc(0U);
  qc.addQubitRegister(1U, "p");
  EXPECT_NO_THROW(qc.addQubitRegister(1U, "q"));
  EXPECT_EQ(qc.getQuantumRegisters().size(), 2U);
}

TEST_F(QFRFunctionality, AddQubitAtFrontOfRegister) {
  QuantumComputation qc{};
  qc.addQubitRegister(2, "q");
  const auto& qregs = qc.getQuantumRegisters();
  EXPECT_EQ(qregs.size(), 1U);
  const auto& reg = qregs.at("q");
  EXPECT_EQ(reg.getStartIndex(), 0U);
  EXPECT_EQ(reg.getSize(), 2U);

  // first remove the qubit
  const auto& [physicalIndex, outputIndex] = qc.removeQubit(0);
  EXPECT_EQ(physicalIndex, 0U);
  EXPECT_EQ(outputIndex, 0U);

  const auto& qregsAfter = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfter.size(), 1U);
  const auto& regAfter = qregsAfter.at("q");
  EXPECT_EQ(regAfter.getStartIndex(), 1U);
  EXPECT_EQ(regAfter.getSize(), 1U);

  // add the qubit back at the front
  qc.addQubit(0, physicalIndex, outputIndex);
  const auto& qregsAfterAdd = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfterAdd.size(), 1U);
  const auto& regAfterAdd = qregsAfterAdd.at("q");
  EXPECT_EQ(regAfterAdd.getStartIndex(), 0U);
  EXPECT_EQ(regAfterAdd.getSize(), 2U);
}

TEST_F(QFRFunctionality, AddQubitAtEndOfRegister) {
  QuantumComputation qc{};
  qc.addQubitRegister(2, "q");
  const auto& qregs = qc.getQuantumRegisters();
  EXPECT_EQ(qregs.size(), 1U);
  const auto& reg = qregs.at("q");
  EXPECT_EQ(reg.getStartIndex(), 0U);
  EXPECT_EQ(reg.getSize(), 2U);

  // first remove the qubit
  const auto& [physicalIndex, outputIndex] = qc.removeQubit(1);
  EXPECT_EQ(physicalIndex, 1U);
  EXPECT_EQ(outputIndex, 1U);

  const auto& qregsAfter = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfter.size(), 1U);
  const auto& regAfter = qregsAfter.at("q");
  EXPECT_EQ(regAfter.getStartIndex(), 0U);
  EXPECT_EQ(regAfter.getSize(), 1U);

  // add the qubit back at the end
  qc.addQubit(1, physicalIndex, outputIndex);
  const auto& qregsAfterAdd = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfterAdd.size(), 1U);
  const auto& regAfterAdd = qregsAfterAdd.at("q");
  EXPECT_EQ(regAfterAdd.getStartIndex(), 0U);
  EXPECT_EQ(regAfterAdd.getSize(), 2U);
}

TEST_F(QFRFunctionality, AddQubitInMiddleOfSplitRegister) {
  QuantumComputation qc{};
  qc.addQubitRegister(3, "q");
  const auto& qregs = qc.getQuantumRegisters();
  EXPECT_EQ(qregs.size(), 1U);
  const auto& reg = qregs.at("q");
  EXPECT_EQ(reg.getStartIndex(), 0U);
  EXPECT_EQ(reg.getSize(), 3U);

  // remove the middle qubit -> splits the register into q_l and q_h
  const auto& [physicalIndex, outputIndex] = qc.removeQubit(1);
  EXPECT_EQ(physicalIndex, 1U);
  EXPECT_EQ(outputIndex, 1U);

  const auto& qregsAfter = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfter.size(), 2U);
  const auto& regLow = qregsAfter.at("q_l");
  EXPECT_EQ(regLow.getStartIndex(), 0U);
  EXPECT_EQ(regLow.getSize(), 1U);
  const auto& regHigh = qregsAfter.at("q_h");
  EXPECT_EQ(regHigh.getStartIndex(), 2U);
  EXPECT_EQ(regHigh.getSize(), 1U);

  // add back the qubit. should consolidate the registers again
  qc.addQubit(1, physicalIndex, outputIndex);
  const auto& qregsAfterAdd = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfterAdd.size(), 1U);
  const auto& regConsolidated = qregsAfterAdd.at("q");
  EXPECT_EQ(regConsolidated.getStartIndex(), 0U);
  EXPECT_EQ(regConsolidated.getSize(), 3U);
}

TEST_F(QFRFunctionality, AddQubitWithoutNeighboringQubits) {
  QuantumComputation qc{};
  qc.addQubitRegister(5, "q");
  const auto& qregs = qc.getQuantumRegisters();
  EXPECT_EQ(qregs.size(), 1U);
  const auto& reg = qregs.at("q");
  EXPECT_EQ(reg.getStartIndex(), 0U);
  EXPECT_EQ(reg.getSize(), 5U);

  // remove the middle 3 qubits -> splits the register into q_l and q_h with one
  // qubit each.
  const auto& [physicalIndex1, outputIndex1] = qc.removeQubit(1);
  EXPECT_EQ(physicalIndex1, 1U);
  EXPECT_EQ(outputIndex1, 1U);
  const auto& [physicalIndex2, outputIndex2] = qc.removeQubit(2);
  EXPECT_EQ(physicalIndex2, 2U);
  EXPECT_EQ(outputIndex2, 2U);
  const auto& [physicalIndex3, outputIndex3] = qc.removeQubit(3);
  EXPECT_EQ(physicalIndex3, 3U);
  EXPECT_EQ(outputIndex3, 3U);

  const auto& qregsAfter = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfter.size(), 2U);
  const auto& regLow = qregsAfter.at("q_l");
  EXPECT_EQ(regLow.getStartIndex(), 0U);
  EXPECT_EQ(regLow.getSize(), 1U);

  const auto& regHigh = qregsAfter.at("q_h");
  EXPECT_EQ(regHigh.getStartIndex(), 4U);
  EXPECT_EQ(regHigh.getSize(), 1U);

  // add back the middle qubit. should create a new register for the qubit
  qc.addQubit(2, physicalIndex2, outputIndex2);
  const auto& qregsAfterAdd = qc.getQuantumRegisters();
  EXPECT_EQ(qregsAfterAdd.size(), 3U);
  // expect to find a `q_2` register with 1 qubit starting at index 2
  const auto& it = qregsAfterAdd.find("q_2");
  ASSERT_NE(it, qregsAfterAdd.end());
  const auto& [nameMiddle, regMiddle] = *it;
  EXPECT_EQ(nameMiddle, "q_2");
  EXPECT_EQ(regMiddle.getStartIndex(), 2U);
  EXPECT_EQ(regMiddle.getSize(), 1U);
}

TEST_F(QFRFunctionality, CopyConstructor) {
  QuantumComputation qc(2, 2);
  qc.h(0);
  qc.swap(0, 1);
  qc.barrier();
  qc.measure(0, 0);
  qc.classicControlled(X, 0, {0, 1});
  const sym::Variable theta{"theta"};
  qc.rx(Symbolic{theta}, 0);

  QuantumComputation compound(2, 2);
  compound.h(0);
  compound.cx(0, 1);
  qc.emplace_back(compound.asOperation());

  qc.initialLayout[0] = 1;
  qc.initialLayout[1] = 0;
  qc.outputPermutation[0] = 1;
  qc.outputPermutation[1] = 0;
  qc.gphase(0.25);

  qc.setLogicalQubitAncillary(1);
  qc.setLogicalQubitGarbage(1);

  const auto qcCopy = qc;
  EXPECT_EQ(qc, qcCopy);
}

TEST_F(QFRFunctionality, InequalityDifferentNumberOfQubits) {
  // Different number of qubits
  const QuantumComputation qc1(2, 2);
  const QuantumComputation qc2(3, 3);
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentInitialLayout) {
  // Different initial layout
  QuantumComputation qc1(2, 2);
  QuantumComputation qc2(2, 2);
  qc1.initialLayout[0] = 0;
  qc1.initialLayout[1] = 1;
  qc2.initialLayout[0] = 1;
  qc2.initialLayout[1] = 0;
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentGateOperations) {
  // Different gate operations
  QuantumComputation qc1(2, 2);
  QuantumComputation qc2(2, 2);
  qc1.h(0);
  qc2.cx(0, 1); // Different operation
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentGateOrder) {
  // Same gates but different order
  QuantumComputation qc1(2, 2);
  QuantumComputation qc2(2, 2);
  qc1.h(0);
  qc1.cx(0, 1);

  qc2.cx(0, 1);
  qc2.h(0); // Reversed order
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentAncillaryQubits) {
  // Different ancillary qubits
  QuantumComputation qc1(2, 2);
  qc1.setLogicalQubitAncillary(1);

  const QuantumComputation qc2(2, 2);
  // No ancillary qubits in qc2
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentGarbageQubits) {
  // Different garbage qubits
  QuantumComputation qc1(2, 2);
  qc1.setLogicalQubitGarbage(1);

  const QuantumComputation qc2(2, 2);
  // No garbage qubits in qc2
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentFinalLayout) {
  // Different final layout
  QuantumComputation qc1(2, 2);
  QuantumComputation qc2(2, 2);
  qc1.outputPermutation[0] = 1;
  qc1.outputPermutation[1] = 0;

  qc2.outputPermutation[0] = 0;
  qc2.outputPermutation[1] = 1;
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentQuantumPhase) {
  // Different quantum phase
  QuantumComputation qc1(2, 2);
  qc1.gphase(0.1); // Add global phase

  const QuantumComputation qc2(2, 2);
  // No global phase in qc2
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentNumberOfClassicalBits) {
  // Different number of classical bits
  const QuantumComputation qc1(2, 3); // 2 qubits, 3 classical bits
  const QuantumComputation qc2(2, 2); // 2 qubits, 2 classical bits
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentMeasurementMappings) {
  // Different measurement mappings
  QuantumComputation qc1(2, 2);
  qc1.measure(0, 1); // Measure qubit 0 to classical bit 1

  QuantumComputation qc2(2, 2);
  qc2.measure(0, 0); // Measure qubit 0 to classical bit 0
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, InequalityDifferentAdditionalOperations) {
  // Different additional operations
  QuantumComputation qc1(2, 2);
  qc1.h(0);

  QuantumComputation qc2(2, 2);
  qc2.h(0);
  qc2.barrier(); // qc2 has an additional barrier
  EXPECT_NE(qc1, qc2);
}

TEST_F(QFRFunctionality, TryAddingExistingQuantumRegister) {
  QuantumComputation qc{};
  qc.addQubitRegister(2, "q");
  EXPECT_THROW(qc.addQubitRegister(2, "q"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryAddingExistingAncillaryRegister) {
  QuantumComputation qc{};
  qc.addAncillaryRegister(2, "a");
  EXPECT_THROW(qc.addAncillaryRegister(2, "a"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryAddingExistingClassicalRegister) {
  QuantumComputation qc{};
  qc.addClassicalRegister(2, "c");
  EXPECT_THROW(qc.addClassicalRegister(2, "c"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryAddingQubitRegisterAfterAncillaryRegister) {
  QuantumComputation qc{};
  qc.addAncillaryRegister(2, "a");
  EXPECT_THROW(qc.addQubitRegister(2, "q"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryAddingZeroSizeQuantumRegister) {
  QuantumComputation qc{};
  EXPECT_THROW(qc.addQubitRegister(0, "q"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryAddingZeroSizeAncillaryRegister) {
  QuantumComputation qc{};
  EXPECT_THROW(qc.addAncillaryRegister(0, "a"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryAddingZeroSizeClassicalRegister) {
  QuantumComputation qc{};
  EXPECT_THROW(qc.addClassicalRegister(0, "c"), std::runtime_error);
}

TEST_F(QFRFunctionality, TryGettingRegisterForQubitNotInRegister) {
  QuantumComputation qc{};
  qc.addQubitRegister(1, "q");
  EXPECT_THROW(std::ignore = qc.getQubitRegister(2), std::runtime_error);
}

TEST_F(QFRFunctionality, stripIdleQubits) {
  QuantumComputation qc(3, 2);
  qc.x(0);
  qc.x(2);
  qc.measure(0, 0);
  qc.measure(2, 1);
  qc.stripIdleQubits(true);

  const auto* const expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q_l;\n"
                               "qubit[1] q_h;\n"
                               "bit[2] c;\n"
                               "x q_l[0];\n"
                               "x q_h[0];\n"
                               "c[0] = measure q_l[0];\n"
                               "c[1] = measure q_h[0];\n";

  EXPECT_EQ(qc.toQASM(), expected);
  const auto qc2 = qasm3::Importer::imports(expected);
  EXPECT_EQ(qc2.toQASM(), expected);
}

TEST_F(QFRFunctionality, failOnAddingExistingQubit) {
  QuantumComputation qc(1);
  EXPECT_THROW(qc.addQubit(0, 0, 0);, std::runtime_error);
  EXPECT_THROW(qc.addAncillaryQubit(0, 0);, std::runtime_error);
}

TEST_F(QFRFunctionality, failOnAddingNonConsecutiveQubit) {
  QuantumComputation qc(1);
  EXPECT_THROW(qc.addQubit(2, 2, 2);, std::runtime_error);
}

TEST_F(QFRFunctionality, failOnGettingIndexOfNonExistingQubit) {
  const QuantumComputation qc(1);
  EXPECT_THROW(std::ignore = qc.getPhysicalQubitIndex(1);, std::runtime_error);
}

TEST_F(QFRFunctionality, failOnRegisterMisconfigurationWithMeasurements) {
  QuantumComputation qc(2);
  qc.addClassicalRegister(1, "c");
  EXPECT_THROW(qc.appendMeasurementsAccordingToOutputPermutation("c");
               , std::runtime_error);
}
} // namespace qc
