/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Expression.hpp"
#include "zx/Rational.hpp"
#include "zx/Utils.hpp"
#include "zx/ZXDefinitions.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>

namespace sym {

using namespace zx;

class ExpressionTest : public ::testing::Test {
public:
  Term<double> x{Variable("x"), 1.0};
  Term<double> y{Variable("y")};
  Term<double> z{Variable("z")};
};

TEST_F(ExpressionTest, basicOps1) {
  Expression<double, PiRational> e(x);

  EXPECT_EQ(1, e.numTerms());
  EXPECT_EQ(PiRational(0, 1), e.getConst());

  e += x;

  EXPECT_EQ(1, e.numTerms());
  EXPECT_EQ(PiRational(0, 1), e.getConst());
  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 2.0);

  e += y;
  EXPECT_EQ(2, e.numTerms());
  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 2.0);
  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[1].getCoeff(), 1.0);
  EXPECT_EQ(e[0].getVar().getName(), "x");
  EXPECT_EQ(e[1].getVar().getName(), "y");
}

TEST_F(ExpressionTest, basicOps2) {
  Expression<double, PiRational> e1;
  e1 += x;
  e1 += 10.0 * y;
  e1 += 5.0 * z;
  e1 += PiRational(1, 2);

  Expression<double, PiRational> e2;
  e2 += -5.0 * x;
  e2 += -10.0 * y;
  e2 += -4.9 * z;
  e2 += PiRational(3, 2);

  auto sum = e1 + e2;

  EXPECT_EQ(2, sum.numTerms());
  EXPECT_PRED_FORMAT2(testing::DoubleLE, sum[0].getCoeff(), -4.0);
  EXPECT_PRED_FORMAT2(testing::DoubleLE, sum[1].getCoeff(), 0.1);
  EXPECT_EQ(sum[0].getVar().getName(), "x");
  EXPECT_EQ(sum[1].getVar().getName(), "z");
  EXPECT_EQ(sum.getConst(), PiRational(0, 1));

  sum += PiRational(1, 1);
  EXPECT_EQ(sum.getConst(), PiRational(1, 1));
  sum -= PiRational(1, 1);
  EXPECT_EQ(sum.getConst(), PiRational(0, 1));
}

TEST_F(ExpressionTest, mult) {
  Expression<double, PiRational> e(x);

  e = e * 2.0;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 2);

  e = e * PiRational(0.5);

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 1);

  e += Expression<double, PiRational>{};

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 1);

  e = e * 0.0;

  EXPECT_TRUE(e.isZero());
}

TEST_F(ExpressionTest, div) {
  Expression<double, Rational> e(x);

  e = e / 2.0;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 0.5);

  e = e / Rational(0.5);

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 1);

  EXPECT_THROW(e = e / 0.0, std::runtime_error);
}

TEST_F(ExpressionTest, Commutativity) {
  const Expression<double, double> e1(x, y);
  Expression<double, double> e2(z);
  e2.setConst(1.0);

  EXPECT_EQ(e1 + e2, e2 + e1);
  EXPECT_EQ(e1 * 2.0, 2.0 * e1);
}

TEST_F(ExpressionTest, Associativity) {
  const Expression<double, double> e1(x, y);
  const Expression<double, double> e2(z);
  const Expression<double, double> e3(1.0);

  EXPECT_EQ(e1 + (e2 + e3), (e1 + e2) + e3);
  EXPECT_EQ(e1 * (2.0 * 4.0), (e1 * 2.0) * 4.0);
}

TEST_F(ExpressionTest, Distributive) {
  const Expression<double, double> e1(x, y);
  const Expression<double, double> e2(z);

  EXPECT_EQ((e1 + e2) * 2.0, (e1 * 2.0) + (e2 * 2.0));
  EXPECT_EQ((e1 - e2) * 2.0, (e1 * 2.0) - (e2 * 2.0));
  std::cout << ((e1 + e2) / 2.0) << "\n";
  std::cout << ((e1 / 2.0) + (e2 / 2.0)) << "\n";
  EXPECT_EQ((e1 + e2) / 2.0, (e1 / 2.0) + (e2 / 2.0));
  EXPECT_EQ((e1 - e2) / 2.0, (e1 / 2.0) - (e2 / 2.0));
}

TEST_F(ExpressionTest, Variable) {
  EXPECT_TRUE(Variable{"x"} != Variable{"y"});
  EXPECT_TRUE(Variable{"x"} == Variable{"x"});
  EXPECT_TRUE(Variable{"x"} < Variable{"y"});
  EXPECT_TRUE(Variable{"z"} > Variable{"y"});
}

TEST_F(ExpressionTest, SumNegation) {
  const Expression<double, double> e1(x, y);
  const Expression<double, double> e2(z, y);

  EXPECT_EQ(e1 - e2, e1 + (-e2));
  const auto& zero = Expression<double, double>{};
  EXPECT_EQ(e1 + (-e1), zero);
}

TEST_F(ExpressionTest, SumMult) {
  const Expression<double, double> e1(x, y);
  EXPECT_EQ(e1 + e1, e1 * 2.0);
}

TEST_F(ExpressionTest, CliffordRounding) {
  constexpr auto eps = 1e-14;
  PiExpression e{PiRational(PI - eps)};
  roundToClifford(e, 1e-9);
  EXPECT_EQ(e, PiExpression(PiRational(1, 1)));
  e = PiExpression{PiRational((PI / 2) - eps)};
  roundToClifford(e, 1e-9);
  EXPECT_EQ(e, PiExpression(PiRational(1, 2)));
  e = PiExpression{PiRational((-PI / 2) - eps)};
  roundToClifford(e, 1e-9);
  EXPECT_EQ(e, PiExpression(PiRational(-1, 2)));
}

TEST_F(ExpressionTest, Clifford) {
  PiExpression e{PiRational(PI)};

  EXPECT_TRUE(isPauli(e));
  EXPECT_TRUE(isClifford(e));
  EXPECT_FALSE(isProperClifford(e));

  e = PiExpression{PiRational(PI / 2)};
  EXPECT_FALSE(isPauli(e));
  EXPECT_TRUE(isClifford(e));
  EXPECT_TRUE(isProperClifford(e));

  e = PiExpression{PiRational(PI / 4)};
  EXPECT_FALSE(isPauli(e));
  EXPECT_FALSE(isClifford(e));
  EXPECT_FALSE(isProperClifford(e));
}

TEST_F(ExpressionTest, Convertability) {
  const Expression<double, double> e({x, y}, PI);
  const auto piE = e.convert<PiRational>();

  EXPECT_EQ(piE, PiExpression({x, y}, PiRational(1, 1)));
}

TEST_F(ExpressionTest, Instantiation) {
  Expression<double, double> e(2 * x, y);

  const VariableAssignment assignment{{Variable{"x"}, 2.0},
                                      {Variable{"y"}, 1.0}};

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e.evaluate(assignment), 5.0);

  e += z;
  EXPECT_THROW([[maybe_unused]] const auto h = e.evaluate(assignment),
               SymbolicException);
}

TEST_F(ExpressionTest, Arithmetic) {
  const auto beta = PiRational(1, 2);

  EXPECT_EQ(beta * 2, PiRational(1, 1));

  const auto betaExpr = PiExpression{beta};
  EXPECT_EQ(betaExpr * 2., PiExpression{PiRational(1, 1)});

  EXPECT_EQ(betaExpr / .5, PiExpression{PiRational(1, 1)});
}
} // namespace sym
