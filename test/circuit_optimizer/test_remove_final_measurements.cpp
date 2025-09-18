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
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "qasm3/Importer.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

namespace qc {
TEST(RemoveFinalMeasurements, removeFinalMeasurements) {
  constexpr std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.h(0);
  qc.h(1);
  qc.measure(0, 0);
  qc.measure(1, 1);
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  ++it;
  ++it; // skip first two H
  const auto& op = *it;
  EXPECT_TRUE(op->isNonUnitaryOperation());
  EXPECT_EQ(op->getType(), qc::Measure);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST(RemoveFinalMeasurements, removeFinalMeasurementsTwoQubitMeasurement) {
  constexpr std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.h(0);
  qc.h(1);
  qc.measure({0, 1}, {0, 1});
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  ++it;
  ++it; // skip first two H
  const auto& op = *it;
  EXPECT_TRUE(op->isNonUnitaryOperation());
  EXPECT_EQ(op->getType(), qc::Measure);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST(RemoveFinalMeasurements, removeFinalMeasurementsCompound) {
  constexpr std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits, nqubits);
  comp.measure(0, 0);
  comp.measure(1, 1);
  comp.h(1);
  qc.emplace_back(comp.asOperation());
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isCompoundOperation());

  const auto* cop = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(cop, nullptr);
  EXPECT_EQ(cop->size(), 2);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST(RemoveFinalMeasurements, removeFinalMeasurementsCompoundDegraded) {
  constexpr std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits, nqubits);
  comp.measure(0, 0);
  comp.h(1);
  qc.emplace_back(comp.asOperation());
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::H);
  EXPECT_EQ(op->getTargets().at(0), 1);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST(RemoveFinalMeasurements, removeFinalMeasurementsCompoundEmpty) {
  constexpr std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits, nqubits);
  comp.measure(0, 0);
  qc.emplace_back(comp.asCompoundOperation());
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::H);
  EXPECT_EQ(op->getTargets().at(0), 1);
}

TEST(RemoveFinalMeasurements, removeFinalMeasurementsWithOperationsInFront) {
  const std::string circ =
      "OPENQASM 2.0;include \"qelib1.inc\";qreg q[3];qreg r[3];h q;cx q, "
      "r;creg c[3];creg d[3];barrier q;measure q->c;measure r->d;\n";
  auto qc = qasm3::Importer::imports(circ);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  ASSERT_EQ(qc.getNops(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 6);
}

TEST(RemoveFinalMeasurements, removeFinalMeasurementsWithBarrier) {
  constexpr std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.barrier({0, 1});
  qc.measure(0, 0);
  qc.measure(1, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_TRUE(qc.empty());
}
} // namespace qc
