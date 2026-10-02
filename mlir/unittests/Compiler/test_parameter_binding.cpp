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

#include <cstdint>
#include <limits>
#include <optional>
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
