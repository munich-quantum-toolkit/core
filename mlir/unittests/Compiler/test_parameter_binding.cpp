/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"

#include "gtest/gtest.h"

#include "mlir/IR/Verifier.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
TEST(ParameterBinding, PartialBindingPreservesRemainingInputs) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%a: f64 {mqt.input_name = "a", mqt.input_id = 7 : i128},
                      %n: i64 {mqt.input_name = "n"},
                      %b: f64 {mqt.input_name = "b", mqt.input_id = 9 : i128})
          attributes {mqt.entry_point} {
        %q = qc.alloc : !qc.qubit
        %sum = arith.addf %a, %b : f64
        qc.ry(%sum) %q : !qc.qubit
        qc.dealloc %q : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"a", "b"}));
  ASSERT_TRUE(qc->bindParameters({{"a", 0.5}}));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"b"}));
  auto entry = mlir::mqt::getEntryPoint(qc->module());
  EXPECT_EQ(entry.getNumArguments(), 2);
  EXPECT_EQ(
      entry.getArgAttrOfType<mlir::IntegerAttr>(1, "mqt.input_id").getInt(), 9);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qco->bindParameters({{"b", -0.5}}));
  EXPECT_TRUE(qco->parameters().empty());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qco->module())));
}

TEST(ParameterBinding, OpenQASMInputsRoundTripThroughQCAndQCO) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(OPENQASM 3.1;
include "stdgates.inc";
input float beta;
input float[64] alpha;
qubit q;
rz(alpha + beta) q;
)");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"beta", "alpha"}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));
  EXPECT_EQ(mlir::mqt::getEntryPoint(qc->module()).getNumResults(), 0);

  auto exported = qc->toOpenQASM3();
  ASSERT_TRUE(exported);
  const auto beta = exported->source().find("input float[64] beta;");
  const auto alpha = exported->source().find("input float[64] alpha;");
  EXPECT_NE(beta, std::string::npos);
  EXPECT_NE(alpha, std::string::npos);
  EXPECT_LT(beta, alpha);
  auto restored = mlir::QCProgram::fromOpenQASMString(exported->source());
  ASSERT_TRUE(restored);
  EXPECT_EQ(restored->parameters(),
            (std::vector<std::string>{"beta", "alpha"}));

  ASSERT_TRUE(restored->bindParameters({{"beta", 0.25}}));
  EXPECT_EQ(restored->parameters(), (std::vector<std::string>{"alpha"}));
  exported = restored->toOpenQASM3();
  ASSERT_TRUE(exported);
  EXPECT_EQ(exported->source().find("input float[64] beta;"),
            std::string::npos);
  EXPECT_NE(exported->source().find("input float[64] alpha;"),
            std::string::npos);

  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(qco);
  ASSERT_TRUE(qco->cleanup());
  EXPECT_EQ(qco->parameters(), (std::vector<std::string>{"alpha"}));
  ASSERT_TRUE(qco->bindParameters({{"alpha", -0.5}}));
  EXPECT_TRUE(qco->parameters().empty());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qco->module())));
}

TEST(ParameterBinding, InvalidBindingDoesNotChangeProgram) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%a: f64 {mqt.input_name = "a"}) attributes {mqt.entry_point} {
        %q = qc.alloc : !qc.qubit
        qc.rx(%a) %q : !qc.qubit
        qc.dealloc %q : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(qc);
  const auto before = qc->str();
  EXPECT_FALSE(qc->bindParameters({{"a", 1.0}, {"unknown", 2.0}}));
  EXPECT_EQ(qc->str(), before);
  EXPECT_FALSE(
      qc->bindParameters({{"a", std::numeric_limits<double>::infinity()}}));
  EXPECT_EQ(qc->str(), before);
}

TEST(ParameterBinding, QIRRequiresBoundParameters) {
  auto qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float theta; qubit q; ry(theta) q;");
  ASSERT_TRUE(qc);
  for (const auto profile :
       {mlir::QIRProfile::Base, mlir::QIRProfile::Adaptive}) {
    auto unbound = qc->copy();
    EXPECT_FALSE(std::move(unbound).intoQIR(profile));
  }
}

TEST(ParameterBinding, OpenQASMRejectsReservedInputName) {
  EXPECT_FALSE(mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float _mqt_theta; qubit q; ry(_mqt_theta) q;"));
}

TEST(ParameterBinding, BindingRejectsReferencedEntryPoint) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%a: f64 {mqt.input_name = "a"}) attributes {mqt.entry_point} {
        return
      }
      func.func @caller(%a: f64) {
        func.call @main(%a) : (f64) -> ()
        return
      }
    }
  )");
  ASSERT_TRUE(qc);
  const auto before = qc->str();
  EXPECT_FALSE(qc->bindParameters({{"a", 1.0}}));
  EXPECT_EQ(qc->str(), before);
}
} // namespace
