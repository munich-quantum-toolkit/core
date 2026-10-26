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
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"

#include "capnp/any.h"
#include "capnp/list.h"
#include "capnp/message.h"
#include "capnp/serialize.h"
#include "gtest/gtest.h"
#include "jeff/Translation/Metadata.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
TEST(ParameterBinding, OpenQASMInputsRoundTripThroughQCAndQCO) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(OPENQASM 3.1;
include "stdgates.inc";
input float beta;
input float[64] alpha;
qubit q;
rz(alpha + beta) q;
)");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"beta", "alpha"}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));
  EXPECT_EQ(mlir::mqt::getEntryPoint(qc->module()).getNumResults(), 0);

  auto exported = std::move(*qc).intoOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(exported));
  const auto beta = exported->source().find("input float[64] beta;");
  const auto alpha = exported->source().find("input float[64] alpha;");
  EXPECT_NE(beta, std::string::npos);
  EXPECT_NE(alpha, std::string::npos);
  EXPECT_LT(beta, alpha);
  auto restored = mlir::QCProgram::fromOpenQASMString(exported->source());
  ASSERT_TRUE(static_cast<bool>(restored))
      << llvm::toString(restored.takeError());
  EXPECT_EQ(restored->parameters(),
            (std::vector<std::string>{"beta", "alpha"}));

  ASSERT_TRUE(mlir::succeeded(restored->bindParameters({{"beta", 0}})));
  EXPECT_EQ(restored->parameters(), (std::vector<std::string>{"alpha"}));
  exported = restored->copy().intoOpenQASM3();
  ASSERT_TRUE(mlir::succeeded(exported));
  EXPECT_EQ(exported->source().find("input float[64] beta;"),
            std::string::npos);
  EXPECT_NE(exported->source().find("input float[64] alpha;"),
            std::string::npos);

  auto qco = std::move(*restored).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qco->cleanup()));
  EXPECT_EQ(qco->parameters(), (std::vector<std::string>{"alpha"}));
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters({{"alpha", 0}})));
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
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
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
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  auto restoredQCO = std::move(*decoded).intoQCO();
  ASSERT_TRUE(mlir::succeeded(restoredQCO));
  auto restoredQC = restoredQCO->copy().intoQC();
  ASSERT_TRUE(mlir::succeeded(restoredQC));
  EXPECT_EQ(restoredQCO->parameters(),
            (std::vector<std::string>{"theta", "phi"}));
  EXPECT_EQ(restoredQC->parameters(),
            (std::vector<std::string>{"theta", "phi"}));
  auto exported = restoredQC->copy().intoOpenQASM3();
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
  exported = restoredQC->copy().intoOpenQASM3();
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
  ASSERT_TRUE(static_cast<bool>(qco)) << llvm::toString(qco.takeError());
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  std::string diagnostic;
  auto context = jeff->context();
  mlir::ScopedDiagnosticHandler handler(context.get(),
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
  ASSERT_TRUE(static_cast<bool>(qco)) << llvm::toString(qco.takeError());
  auto jeff = std::move(*qco).intoJeff();
  ASSERT_TRUE(mlir::succeeded(jeff));
  capnp::MallocMessageBuilder message;
  auto list =
      message.getRoot<capnp::AnyPointer>().initAs<capnp::List<uint32_t>>(1);
  list.set(0, 42);
  auto words = capnp::messageToFlatArray(message);
  auto bytes = words.asBytes();
  auto context = jeff->context();
  mlir::Builder builder(context.get());
  auto metadata = builder.getDictionaryAttr({
      builder.getNamedAttr("name", builder.getStringAttr("mqt.input_name")),
      builder.getNamedAttr(
          "value",
          builder.getDenseI8ArrayAttr(llvm::ArrayRef(
              reinterpret_cast<const int8_t*>(bytes.begin()), bytes.size()))),
  });
  auto function = jeff->module().lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(mlir::succeeded(mlir::jeff::setMetadata(
      function.getArgument(0), builder.getArrayAttr({metadata}))));
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(context.get(),
                                        [&](mlir::Diagnostic& error) {
                                          diagnostic += error.str();
                                          return mlir::success();
                                        });
  EXPECT_TRUE(mlir::failed(std::move(*jeff).intoQCO()));
  EXPECT_NE(diagnostic.find("failed to decode input name metadata"),
            std::string::npos)
      << diagnostic;
}

TEST(ParameterBinding, MixedArrayBindingPreservesRemainingInputs) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%a: tensor<2xf64> {mqt.input_name = "a"},
                      %theta: f64 {mqt.input_name = "theta", mqt.input_id = 7 : i128},
                      %b: tensor<1xf64> {mqt.input_name = "b", mqt.input_id = 9 : i128})
          attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %a0 = tensor.extract %a[%zero] : tensor<2xf64>
        %a1 = tensor.extract %a[%one] : tensor<2xf64>
        %b0 = tensor.extract %b[%zero] : tensor<1xf64>
        %sum = arith.addf %a0, %a1 : f64
        %offset = arith.addf %sum, %theta : f64
        %angle = arith.addf %offset, %b0 : f64
        %q = qc.alloc : !qc.qubit
        qc.ry(%angle) %q : !qc.qubit
        qc.dealloc %q : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"a", "theta", "b"}));
  ASSERT_TRUE(mlir::succeeded(
      qc->bindParameters({{"a", std::vector<double>{0.25, -0.5}}})));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"theta", "b"}));
  auto entry = mlir::mqt::getEntryPoint(qc->module());
  EXPECT_EQ(
      entry.getArgAttrOfType<mlir::IntegerAttr>(0, "mqt.input_id").getInt(), 7);
  EXPECT_EQ(
      entry.getArgAttrOfType<mlir::IntegerAttr>(1, "mqt.input_id").getInt(), 9);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));

  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  const std::map<std::string, double> scalarValues{{"theta", 0.625}};
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters(scalarValues)));
  EXPECT_EQ(qco->parameters(), (std::vector<std::string>{"b"}));
  ASSERT_TRUE(
      mlir::succeeded(qco->bindParameters({{"b", std::vector<double>{-0.5}}})));
  EXPECT_TRUE(qco->parameters().empty());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qco->module())));
  auto rotations =
      mlir::mqt::getEntryPoint(qco->module()).getOps<mlir::qco::RYOp>();
  ASSERT_FALSE(rotations.empty());
  auto rotation = *rotations.begin();
  auto angle =
      rotation.getParameter(0).getDefiningOp<mlir::arith::ConstantOp>();
  ASSERT_TRUE(angle);
  EXPECT_EQ(llvm::cast<mlir::FloatAttr>(angle.getValue()).getValueAsDouble(),
            -0.125);
  auto restored = std::move(*qco).intoQC();
  ASSERT_TRUE(mlir::succeeded(restored));
  for (const auto profile :
       {mlir::QIRProfile::Base, mlir::QIRProfile::Adaptive}) {
    auto bound = restored->copy();
    EXPECT_TRUE(mlir::succeeded(std::move(bound).intoQIR(profile)));
  }
  EXPECT_TRUE(mlir::succeeded(std::move(*restored).intoOpenQASM3()));
}

TEST(ParameterBinding, OnlyNamedScalarsAndStaticRankOneF64InputsAreBindable) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%scalar: f64 {mqt.input_name = "scalar"},
                      %n: i64 {mqt.input_name = "n"},
                      %array: tensor<2xf64> {mqt.input_name = "array", mqt.input_id = 9 : i128},
                      %empty: tensor<0xf64> {mqt.input_name = "empty"},
                      %dynamic: tensor<?xf64> {mqt.input_name = "dynamic"},
                      %matrix: tensor<2x2xf64> {mqt.input_name = "matrix"},
                      %rank_zero: tensor<f64> {mqt.input_name = "rank_zero"},
                      %f32: tensor<2xf32> {mqt.input_name = "f32"},
                      %vector: vector<2xf64> {mqt.input_name = "vector"},
                      %unnamed: tensor<2xf64>)
          attributes {mqt.entry_point} {
        return
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  EXPECT_EQ(qc->parameters(),
            (std::vector<std::string>{"scalar", "array", "empty"}));
  const std::map<std::string, double> values{{"scalar", 0.5}};
  ASSERT_TRUE(mlir::succeeded(qc->bindParameters(values)));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"array", "empty"}));
  auto entry = mlir::mqt::getEntryPoint(qc->module());
  EXPECT_TRUE(entry.getArgument(0).getType().isInteger(64));
  EXPECT_EQ(
      entry.getArgAttrOfType<mlir::IntegerAttr>(1, "mqt.input_id").getInt(), 9);
  ASSERT_TRUE(
      mlir::succeeded(qc->bindParameters({{"empty", std::vector<double>{}}})));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"array"}));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(qc->module())));
}

TEST(ParameterBinding, InvalidBindingDoesNotChangeProgram) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%a: f64 {mqt.input_name = "a"},
                      %b: tensor<2xf64> {mqt.input_name = "b"})
          attributes {mqt.entry_point} {
        return
      }
    }
  )");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  const auto before = qc->str();
  for (const auto& invalid : std::vector<mlir::ParameterValue>{
           1.0,
           std::vector<double>{},
           std::vector<double>{0.5},
           std::vector<double>{0.5, 0.5, 0.5},
           std::vector<double>{0.5, std::numeric_limits<double>::quiet_NaN()},
           std::vector<double>{std::numeric_limits<double>::infinity(), 0.5},
       }) {
    EXPECT_TRUE(mlir::failed(qc->bindParameters({{"a", 0.5}, {"b", invalid}})));
    EXPECT_EQ(qc->str(), before);
  }
  for (const auto& invalid : std::vector<mlir::ParameterValue>{
           std::vector<double>{0.5},
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
       }) {
    EXPECT_TRUE(mlir::failed(qc->bindParameters({{"a", invalid}})));
    EXPECT_EQ(qc->str(), before);
  }
  EXPECT_TRUE(mlir::failed(qc->bindParameters(
      {{"b", std::vector<double>{0.5, 0.5}}, {"unknown", 0.0}})));
  EXPECT_EQ(qc->str(), before);
  EXPECT_TRUE(mlir::succeeded(qc->bindParameters({})));
  EXPECT_EQ(qc->str(), before);
}

TEST(ParameterBinding, RejectsReachableNonFiniteGateParameters) {
  auto qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float theta; qubit q; ry(1.0 / theta) q;");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  EXPECT_TRUE(mlir::failed(qc->bindParameters({{"theta", 0.0}})));

  qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float theta; input float unused; qubit q; "
      "if (theta != 0) { ry(1.0 / theta) q; } else { x q; } "
      "output bit result; result = measure q;");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  auto qco = qc->copy().intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  ASSERT_TRUE(mlir::succeeded(qc->bindParameters({{"theta", 0.0}})));
  EXPECT_EQ(qc->parameters(), (std::vector<std::string>{"unused"}));
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters({{"theta", 0.0}})));
  EXPECT_EQ(qco->parameters(), (std::vector<std::string>{"unused"}));
  ASSERT_TRUE(mlir::succeeded(qco->bindParameters({{"unused", 1.0}})));
  const auto samples =
      mlir::qco::sample(mlir::mqt::getEntryPoint(qco->module()), 4, 42);
  ASSERT_TRUE(mlir::succeeded(samples));
  EXPECT_EQ(samples->at("1"), 4);
}

TEST(ParameterBinding, QIRRequiresBoundParameters) {
  auto qc = mlir::QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; input float theta; qubit q; ry(theta) q;");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  for (const auto profile :
       {mlir::QIRProfile::Base, mlir::QIRProfile::Adaptive}) {
    auto unbound = qc->copy();
    EXPECT_TRUE(mlir::failed(std::move(unbound).intoQIR(profile)));
  }
}

TEST(ParameterBinding, OpenQASMRejectsReservedInputName) {
  EXPECT_TRUE(llvm::errorToBool(
      (mlir::QCProgram::fromOpenQASMString(
           "OPENQASM 3.1; input float _mqt_theta; qubit q; ry(_mqt_theta) q;"))
          .takeError()));
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
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  const auto before = qc->str();
  EXPECT_TRUE(mlir::failed(qc->bindParameters({{"a", 1.0}})));
  EXPECT_EQ(qc->str(), before);
}
} // namespace
