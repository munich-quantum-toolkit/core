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
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <gtest/gtest.h>

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
  qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_FALSE(ifElse.isUnitary());
}

TEST(IfElseOperation, IsNonUnitaryOperation) {
  qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_TRUE(ifElse.isNonUnitaryOperation());
}

TEST(IfElseOperation, IsIfElseOperation) {
  qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_TRUE(ifElse.isIfElseOperation());
}

TEST(IfElseOperation, IsControlled) {
  qc::IfElseOperation ifElse(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0);

  ASSERT_FALSE(ifElse.isControlled());
}

TEST(IfElseOperation, Equals) {
  qc::IfElseOperation ifElse1(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse2(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse3(
      std::make_unique<qc::StandardOperation>(1, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(0, qc::OpType::Y), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse4(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Z), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse5(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse6(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, 0U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse7(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, 1U,
      qc::ComparisonKind::Neq);

  EXPECT_TRUE(ifElse1.equals(ifElse2));
  EXPECT_FALSE(ifElse1.equals(ifElse3));
  EXPECT_FALSE(ifElse1.equals(ifElse4));
  EXPECT_FALSE(ifElse1.equals(ifElse5));
  EXPECT_FALSE(ifElse1.equals(ifElse6));
  EXPECT_FALSE(ifElse1.equals(ifElse7));
}

TEST(IfElseOperation, Hash) {
  qc::IfElseOperation ifElse1(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse2(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse3(
      std::make_unique<qc::StandardOperation>(1, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(0, qc::OpType::Y), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse4(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Z), 0, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse5(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, 1U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse6(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, 0U,
      qc::ComparisonKind::Eq);
  qc::IfElseOperation ifElse7(
      std::make_unique<qc::StandardOperation>(0, qc::OpType::X),
      std::make_unique<qc::StandardOperation>(1, qc::OpType::Y), 1, 1U,
      qc::ComparisonKind::Neq);

  std::hash<qc::IfElseOperation> hasher;

  EXPECT_EQ(hasher(ifElse1), hasher(ifElse2));
  EXPECT_NE(hasher(ifElse1), hasher(ifElse3));
  EXPECT_NE(hasher(ifElse1), hasher(ifElse4));
  EXPECT_NE(hasher(ifElse1), hasher(ifElse5));
  EXPECT_NE(hasher(ifElse1), hasher(ifElse6));
  EXPECT_NE(hasher(ifElse1), hasher(ifElse7));
}
