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
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

TEST(IfElseOperation, GetInvertedComparisonKind) {
  EXPECT_EQ(qc::getInvertedComparisonKind(qc::ComparisonKind::Lt),
            qc::ComparisonKind::Geq);
  EXPECT_EQ(qc::getInvertedComparisonKind(qc::ComparisonKind::Leq),
            qc::ComparisonKind::Gt);
  EXPECT_EQ(qc::getInvertedComparisonKind(qc::ComparisonKind::Gt),
            qc::ComparisonKind::Leq);
  EXPECT_EQ(qc::getInvertedComparisonKind(qc::ComparisonKind::Geq),
            qc::ComparisonKind::Lt);
  EXPECT_EQ(qc::getInvertedComparisonKind(qc::ComparisonKind::Eq),
            qc::ComparisonKind::Neq);
  EXPECT_EQ(qc::getInvertedComparisonKind(qc::ComparisonKind::Neq),
            qc::ComparisonKind::Eq);
}

TEST(IfElseOperation, Assignment) {
  const qc::IfElseOperation ifElse1(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);
  qc::IfElseOperation ifElse2(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::Y),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Z), 0);

  ifElse2 = ifElse1;

  EXPECT_TRUE(ifElse2.equals(ifElse1));

  // Check that operations have been cloned
  EXPECT_NE(ifElse2.getThenOp(), ifElse1.getThenOp());
  EXPECT_NE(ifElse2.getElseOp(), ifElse1.getElseOp());
}

TEST(IfElseOperation, Apply) {
  qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  qc::Permutation permutation{};
  permutation[0] = 1;
  permutation[1] = 0;

  ifElse.apply(permutation);

  EXPECT_EQ(ifElse.getThenOp()->getTargets().at(0), static_cast<qc::Qubit>(1));
  EXPECT_EQ(ifElse.getElseOp()->getTargets().at(0), static_cast<qc::Qubit>(0));
}

TEST(IfElseOperation, IsUnitary) {
  const qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_FALSE(ifElse.isUnitary());
}

TEST(IfElseOperation, IsNonUnitaryOperation) {
  const qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_TRUE(ifElse.isNonUnitaryOperation());
}

TEST(IfElseOperation, IsIfElseOperation) {
  const qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_TRUE(ifElse.isIfElseOperation());
}

TEST(IfElseOperation, IsControlled) {
  const qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_FALSE(ifElse.isControlled());
}

TEST(IfElseOperation, Equals) {
  const qc::IfElseOperation ifElseBit1(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0, true,
      qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit2(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0, true,
      qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit3(
      std::make_unique<qc::StandardOperation>(1, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(0, qc::OpType::Y), 0, true,
      qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit4(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Z), 0, true,
      qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit5(
      nullptr, std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0,
      true, qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit6(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X), nullptr, 0,
      true, qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit7(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, true,
      qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit8(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, false,
      qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseBit9(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, true,
      qc::ComparisonKind::Neq);

  const qc::ClassicalRegister controlRegister1(0, 1);
  const qc::ClassicalRegister controlRegister2(0, 2);

  const qc::IfElseOperation ifElseRegister1(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y),
      controlRegister1, 1U, qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseRegister2(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y),
      controlRegister1, 1U, qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseRegister3(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y),
      controlRegister2, 1U, qc::ComparisonKind::Eq);
  const qc::IfElseOperation ifElseRegister4(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y),
      controlRegister1, 2U, qc::ComparisonKind::Eq);

  EXPECT_TRUE(ifElseBit1.equals(ifElseBit2));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit3));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit4));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit5));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit6));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit7));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit8));
  EXPECT_FALSE(ifElseBit1.equals(ifElseBit9));

  EXPECT_FALSE(ifElseBit1.equals(ifElseRegister1));

  EXPECT_TRUE(ifElseRegister1.equals(ifElseRegister2));
  EXPECT_FALSE(ifElseRegister1.equals(ifElseRegister3));
  EXPECT_FALSE(ifElseRegister1.equals(ifElseRegister4));

  const std::hash<qc::IfElseOperation> hasher;

  EXPECT_EQ(hasher(ifElseBit1), hasher(ifElseBit2));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit3));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit4));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit5));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit6));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit7));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit8));
  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseBit9));

  EXPECT_NE(hasher(ifElseBit1), hasher(ifElseRegister1));

  EXPECT_EQ(hasher(ifElseRegister1), hasher(ifElseRegister2));
  EXPECT_NE(hasher(ifElseRegister1), hasher(ifElseRegister3));
  EXPECT_NE(hasher(ifElseRegister1), hasher(ifElseRegister4));
}

TEST(IfElseOperation, RuntimeErrors) {
  qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  EXPECT_THROW(ifElse.invert(), std::runtime_error);
  EXPECT_THROW(ifElse.setTargets({}), std::runtime_error);
  EXPECT_THROW(ifElse.setControls({}), std::runtime_error);
  EXPECT_THROW(ifElse.addControl(qc::Control(0)), std::runtime_error);
  EXPECT_THROW(ifElse.clearControls(), std::runtime_error);
  EXPECT_THROW(ifElse.removeControl(qc::Control(0)), std::runtime_error);
  EXPECT_THROW(ifElse.setGate(qc::OpType::None), std::runtime_error);
  EXPECT_THROW(ifElse.setParameter({}), std::runtime_error);
}
