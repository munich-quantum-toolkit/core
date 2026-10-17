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

#include "capnp/any.h"
#include "capnp/list.h"
#include "capnp/message.h"
#include "capnp/serialize.h"
#include "gtest/gtest.h"
#include "jeff/Translation/Metadata.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
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
  ASSERT_TRUE(mlir::succeeded(qc));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"a", "b"}));
  ASSERT_TRUE(mlir::succeeded(qc->bindParameters({{"a", 0.5}})));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"b"}));
  auto entry = mlir::mqt::getEntryPoint(qc->module());
  EXPECT_EQ(entry.getNumArguments(), 2);
  EXPECT_EQ(
      entry.getArgAttrOfType<mlir::IntegerAttr>(1, "mqt.input_id").getInt(), 9);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters({{"b", -0.5}})));
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
  ASSERT_TRUE(mlir::succeeded(qc));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"beta", "alpha"}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));
  EXPECT_EQ(mlir::mqt::getEntryPoint(qc->module()).getNumResults(), 0);

  auto exported = qc->toOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(exported));
  const auto beta = exported->source().find("input float[64] beta;");
  const auto alpha = exported->source().find("input float[64] alpha;");
  EXPECT_NE(beta, std::string::npos);
  EXPECT_NE(alpha, std::string::npos);
  EXPECT_LT(beta, alpha);
  auto restored = mlir::QCProgram::fromOpenQASMString(exported->source());
  ASSERT_TRUE(mlir::succeeded(restored));
  EXPECT_EQ(restored->parameters(),
            (std::vector<std::string>{"beta", "alpha"}));

  ASSERT_TRUE(mlir::succeeded(restored->bindParameters({{"beta", 0.25}})));
  EXPECT_EQ(restored->parameters(), (std::vector<std::string>{"alpha"}));
  exported = restored->toOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(exported));
  EXPECT_EQ(exported->source().find("input float[64] beta;"),
            std::string::npos);
  EXPECT_NE(exported->source().find("input float[64] alpha;"),
            std::string::npos);

  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qco->cleanup()));
  EXPECT_EQ(qco->parameters(), (std::vector<std::string>{"alpha"}));
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters({{"alpha", -0.5}})));
  EXPECT_TRUE(qco->parameters().empty());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qco->module())));
}

TEST(ParameterBinding, NamedInputsSurviveJeffSerialization) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(OPENQASM 3.1;
include "stdgates.inc";
input float theta;
input float phi;
qubit q;
rx(theta) q;
rz(phi) q;
)");
  ASSERT_TRUE(mlir::succeeded(qc));
  auto entry = mlir::mqt::getEntryPoint(qc->module());
  entry.setArgAttr(
      0, "mqt.input_id",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(qc->module().getContext(), 128), 7));
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  auto decoded = mlir::JeffProgram::fromBytes(jeff->toBytes());
  ASSERT_TRUE(mlir::succeeded(decoded));
  auto restoredQCO = std::move(*decoded).intoQCO();
  ASSERT_TRUE(mlir::succeeded(restoredQCO));
  auto restoredQC = restoredQCO->copy().intoQC();
  ASSERT_TRUE(mlir::succeeded(restoredQC));
  EXPECT_EQ(restoredQCO->parameters(),
            (std::vector<std::string>{"theta", "phi"}));
  EXPECT_EQ(restoredQC->parameters(),
            (std::vector<std::string>{"theta", "phi"}));
  auto exported = restoredQC->toOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(exported));
  EXPECT_NE(exported->source().find("input float[64] theta;"),
            std::string::npos);
  EXPECT_NE(exported->source().find("input float[64] phi;"), std::string::npos);

  ASSERT_TRUE(mlir::succeeded(restoredQCO->bindParameters({{"theta", 0.25}})));
  ASSERT_TRUE(mlir::succeeded(restoredQC->bindParameters({{"theta", 0.25}})));
  EXPECT_EQ(restoredQCO->parameters(), (std::vector<std::string>{"phi"}));
  EXPECT_EQ(restoredQC->parameters(), (std::vector<std::string>{"phi"}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(restoredQCO->module())));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(restoredQC->module())));
  exported = restoredQC->toOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(exported));
  EXPECT_EQ(exported->source().find("input float[64] theta;"),
            std::string::npos);
  EXPECT_NE(exported->source().find("input float[64] phi;"), std::string::npos);
}

TEST(ParameterBinding, RejectsMalformedJeffInputNameMetadata) {
  auto qco = mlir::QCOProgram::fromMLIRString(R"(
    module {
      func.func @main(%theta: f64 {jeff.metadata = [
        {name = "mqt.input_name", value = array<i8: 0>}
      ]}) attributes {mqt.entry_point} {
        return
      }
    }
  )");
  ASSERT_TRUE(mlir::succeeded(qco));
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(jeff->module().getContext(),
                                        [&](mlir::Diagnostic& error) {
                                          diagnostic += error.str();
                                          return mlir::success();
                                        });
  EXPECT_TRUE(mlir::failed(std::move(*jeff).intoQCO()));
  EXPECT_NE(diagnostic.find("Metadata payload size must be a multiple"),
            std::string::npos)
      << diagnostic;
}

TEST(ParameterBinding, RejectsNonTextJeffInputNameMetadata) {
  auto qco = mlir::QCOProgram::fromMLIRString(R"(
    module {
      func.func @main(%theta: f64) attributes {mqt.entry_point} {
        return
      }
    }
  )");
  ASSERT_TRUE(mlir::succeeded(qco));
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  capnp::MallocMessageBuilder message;
  auto list =
      message.getRoot<capnp::AnyPointer>().initAs<capnp::List<uint32_t>>(1);
  list.set(0, 42);
  auto words = capnp::messageToFlatArray(message);
  auto bytes = words.asBytes();
  auto* context = jeff->module().getContext();
  auto metadata = mlir::DictionaryAttr::get(
      context,
      {
          {
              mlir::StringAttr::get(context, "name"),
              mlir::StringAttr::get(context, "mqt.input_name"),
          },
          {
              mlir::StringAttr::get(context, "value"),
              mlir::DenseI8ArrayAttr::get(
                  context,
                  llvm::ArrayRef(reinterpret_cast<const int8_t*>(bytes.begin()),
                                 bytes.size())),
          },
      });
  auto function = jeff->module().lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(
      function.getArgument(0), mlir::ArrayAttr::get(context, {metadata}))));
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(context, [&](mlir::Diagnostic& error) {
    diagnostic += error.str();
    return mlir::success();
  });
  EXPECT_TRUE(mlir::failed(std::move(*jeff).intoQCO()));
  EXPECT_NE(diagnostic.find("failed to decode input name metadata"),
            std::string::npos)
      << diagnostic;
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
  ASSERT_TRUE(mlir::succeeded(qc));
  const auto before = qc->str();
  EXPECT_TRUE(mlir::failed(qc->bindParameters({{"a", 1.0}, {"unknown", 2.0}})));
  EXPECT_EQ(qc->str(), before);
  EXPECT_TRUE(mlir::failed(
      qc->bindParameters({{"a", std::numeric_limits<double>::infinity()}})));
  EXPECT_EQ(qc->str(), before);
}

TEST(ParameterBinding, QIRRequiresBoundParameters) {
  auto qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float theta; qubit q; ry(theta) q;");
  ASSERT_TRUE(mlir::succeeded(qc));
  for (const auto profile :
       {mlir::QIRProfile::Base, mlir::QIRProfile::Adaptive}) {
    auto unbound = qc->copy();
    EXPECT_TRUE(mlir::failed(std::move(unbound).intoQIR(profile)));
  }
}

TEST(ParameterBinding, OpenQASMRejectsReservedInputName) {
  EXPECT_TRUE(mlir::failed(mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float _mqt_theta; qubit q; ry(_mqt_theta) q;")));
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
  ASSERT_TRUE(mlir::succeeded(qc));
  const auto before = qc->str();
  EXPECT_TRUE(mlir::failed(qc->bindParameters({{"a", 1.0}})));
  EXPECT_EQ(qc->str(), before);
}
} // namespace
