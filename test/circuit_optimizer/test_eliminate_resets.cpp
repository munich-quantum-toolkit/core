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
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>

namespace qc {
TEST(EliminateResets, basicTest) {
  QuantumComputation qc{};
  qc.addQubitRegister(1);
  qc.addClassicalRegister(2);
  qc.h(0);
  qc.measure(0, 0U);
  qc.reset(0);
  qc.h(0);
  qc.measure(0, 1U);

  std::cout << qc << "\n";

  EXPECT_TRUE(qc.isDynamic());

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 4);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);
  const auto& op3 = qc.at(3);

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(0));
  const auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_TRUE(op2->getType() == qc::H);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  EXPECT_TRUE(op2->getControls().empty());

  EXPECT_TRUE(op3->getType() == qc::Measure);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(1));
  auto* measure1 = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
  ASSERT_NE(measure1, nullptr);
  const auto& classics1 = measure1->getClassics();
  EXPECT_EQ(classics1.size(), 1);
  EXPECT_EQ(classics1.at(0), 1);
}

TEST(EliminateResets, testIf) {
  QuantumComputation qc{};
  qc.addQubitRegister(1);
  qc.addClassicalRegister(2);
  qc.h(0);
  qc.measure(0, 0U);
  qc.reset(0);
  qc.if_(X, 0, 0);

  std::cout << qc << "\n";

  EXPECT_TRUE(qc.isDynamic());

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 3);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_TRUE(op2->isIfElseOperation());
  auto* ifElse = dynamic_cast<qc::IfElseOperation*>(op2.get());
  ASSERT_NE(ifElse, nullptr);
  const auto& thenOp = ifElse->getThenOp();
  EXPECT_TRUE(thenOp->getType() == qc::X);
  EXPECT_EQ(thenOp->getNtargets(), 1);
  const auto& targets = thenOp->getTargets();
  EXPECT_EQ(targets.at(0), 1);
  EXPECT_EQ(thenOp->getNcontrols(), 0);
}

TEST(EliminateResets, testIfElse) {
  QuantumComputation qc{};
  qc.addQubitRegister(1);
  qc.addClassicalRegister(2);
  qc.h(0);
  qc.measure(0, 0U);
  qc.reset(0);
  qc.ifElse(std::make_unique<StandardOperation>(0, X),
            std::make_unique<StandardOperation>(0, Y), 0);

  std::cout << qc << "\n";

  EXPECT_TRUE(qc.isDynamic());

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 3);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_TRUE(op2->isIfElseOperation());
  auto* ifElse = dynamic_cast<qc::IfElseOperation*>(op2.get());
  ASSERT_NE(ifElse, nullptr);
  const auto& thenOp = ifElse->getThenOp();
  EXPECT_TRUE(thenOp->getType() == qc::X);
  EXPECT_EQ(thenOp->getNtargets(), 1);
  const auto& thenTargets = thenOp->getTargets();
  EXPECT_EQ(thenTargets.at(0), 1);
  EXPECT_EQ(thenOp->getNcontrols(), 0);
  const auto& elseOp = ifElse->getElseOp();
  EXPECT_TRUE(elseOp->getType() == qc::Y);
  EXPECT_EQ(elseOp->getNtargets(), 1);
  const auto& elseTargets = elseOp->getTargets();
  EXPECT_EQ(elseTargets.at(0), 1);
  EXPECT_EQ(elseOp->getNcontrols(), 0);
}

TEST(EliminateResets, testMultipleTargetReset) {
  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.reset({0, 1});
  qc.x(0);
  qc.z(1);
  qc.cx(1, 0);

  std::cout << qc << "\n";

  EXPECT_TRUE(qc.isDynamic());

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 4);
  ASSERT_EQ(qc.getNindividualOps(), 3);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);

  EXPECT_TRUE(op0->getType() == qc::X);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(2));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::Z);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(3));
  EXPECT_TRUE(op1->getControls().empty());

  EXPECT_TRUE(op2->getType() == qc::X);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(2));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(3), 1);
}

TEST(EliminateResets, testCompoundOperation) {
  QuantumComputation qc(2U, 2U);

  qc.reset(0);
  qc.reset(1);

  QuantumComputation comp(2U, 2U);
  comp.cx(1, 0);
  comp.reset(0);
  comp.measure(0, 0);
  comp.if_(X, 0, 0);
  qc.emplace_back(comp.asOperation());

  std::cout << qc << "\n";

  EXPECT_TRUE(qc.isDynamic());

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 5);
  ASSERT_EQ(qc.getNindividualOps(), 3);

  const auto& op = qc.at(0);
  EXPECT_TRUE(op->isCompoundOperation());
  auto* compOp0 = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(compOp0, nullptr);
  EXPECT_EQ(compOp0->size(), 3);

  const auto& op0 = compOp0->at(0);
  const auto& op1 = compOp0->at(1);
  const auto& op2 = compOp0->at(2);

  EXPECT_TRUE(op0->getType() == qc::X);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(2));
  const auto& controls0 = op0->getControls();
  EXPECT_EQ(controls0.size(), 1);
  EXPECT_EQ(controls0.count(3), 1);

  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(4));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_TRUE(op2->isIfElseOperation());
  auto* ifElse = dynamic_cast<qc::IfElseOperation*>(op2.get());
  ASSERT_NE(ifElse, nullptr);
  const auto& thenOp = ifElse->getThenOp();
  EXPECT_TRUE(thenOp->getType() == qc::X);
  EXPECT_EQ(thenOp->getNtargets(), 1);
  const auto& targets = thenOp->getTargets();
  EXPECT_EQ(targets.at(0), 4);
  EXPECT_EQ(thenOp->getNcontrols(), 0);
}
} // namespace qc
