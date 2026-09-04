/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file test_mqt_ir.cpp
 * @brief Unit tests for the MQT metadata dialect.
 */

#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTAttributes.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <memory>
#include <string>

using namespace mlir;

namespace {
class MQTIRTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect,
                    memref::MemRefDialect, mqt::MQTDialect, qc::QCDialect,
                    qco::QCODialect, qtensor::QTensorDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(const StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  [[nodiscard]] Attribute parseAttr(const StringRef source) const {
    return parseAttribute(source, context.get());
  }

  [[nodiscard]] Attribute roundTrip(const Attribute attribute) const {
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    attribute.print(stream);
    return parseAttr(printed);
  }
};

TEST_F(MQTIRTest, AcceptsProgramInputAndRegisterNames) {
  EXPECT_TRUE(parse(R"mlir(
    module {
      func.func @qc(%theta: f64 {mqt.input_name = "theta[2]",
          mqt.parameter_group = {identity = "group-id", name = "theta",
                                 index = 2 : i64, size = 4 : i64}},
          %element: f64 {mqt.input_name = "[0]",
          mqt.parameter_group = {identity = "empty-name", name = "",
                                 index = 0 : i64, size = 1 : i64}}) {
        %reg = memref.alloc() {mqt.register_name = "q"}
            : memref<2x!qc.qubit>
        return
      }
      func.func @qco(%enabled: i1 {mqt.input_name = "enabled"}) {
        %c2 = arith.constant 2 : index
        %reg = qtensor.alloc(%c2) {mqt.register_name = "r"}
            : tensor<2x!qco.qubit>
        return
      }
      func.func @cbit() {
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"}
            : !cbit.reg<2>
        return
      }
      func.func @lowered_cbit() {
        %reg = memref.alloc() {mqt.register_name = "lowered"} : memref<2xi1>
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RoundTripsTypedCompilationTarget) {
  const auto compilationTarget =
      dyn_cast_if_present<mqt::CompilationTargetAttr>(
          parseAttr(R"mlir(#mqt.compilation_target<
            name = "device",
            sites = [<id = 4>, <id = 7>],
            connectivity = explicit,
            couplings = [<source = 4, target = 7>],
            native_operations = explicit,
            operations = [
                <name = "cx",
                    arity = #mqt.operation_arity<kind = fixed, value = 2>,
                    num_parameters = 0,
                    site_tuples = [<[4, 7], fidelity = 9.900000e-01 : f64>]>,
                <name = "gphase",
                    arity = #mqt.operation_arity<kind = fixed, value = 0>,
                    num_parameters = 1, site_tuples = []>,
                <name = "h",
                    arity = #mqt.operation_arity<kind = variadic, value = 1>,
                    num_parameters = 0, site_tuples = []>]>)mlir"));
  ASSERT_TRUE(compilationTarget);
  EXPECT_EQ(compilationTarget.getName().getValue(), "device");
  ASSERT_EQ(compilationTarget.getSites().size(), 2U);
  EXPECT_EQ(compilationTarget.getSites()[0].getId(), 4);
  EXPECT_EQ(compilationTarget.getSites()[1].getId(), 7);
  EXPECT_EQ(compilationTarget.getConnectivity(),
            mqt::ConnectivityKind::Explicit);
  EXPECT_EQ(compilationTarget.getNativeOperations(),
            mqt::NativeOperationsKind::Explicit);
  ASSERT_EQ(compilationTarget.getOperations().size(), 3U);
  EXPECT_EQ(compilationTarget.getOperations()[0].getArity().getKind(),
            mqt::OperationArityKind::Fixed);
  EXPECT_EQ(compilationTarget.getOperations()[1].getArity().getValue(), 0U);
  EXPECT_EQ(compilationTarget.getOperations()[2].getArity().getKind(),
            mqt::OperationArityKind::Variadic);
  ASSERT_EQ(compilationTarget.getOperations()[0].getSiteTuples().size(), 1U);
  const auto tuple = compilationTarget.getOperations()[0].getSiteTuples()[0];
  EXPECT_EQ(tuple.getSites(), (ArrayRef<int64_t>{4, 7}));
  EXPECT_EQ(tuple.getFidelity().getValueAsDouble(), 0.99);
  EXPECT_TRUE(compilationTarget.getOperations()[1].getSiteTuples().empty());
  EXPECT_TRUE(compilationTarget.getOperations()[2].getSiteTuples().empty());

  EXPECT_EQ(roundTrip(compilationTarget), compilationTarget);
}

TEST_F(MQTIRTest, RoundTripsMaximumSiteIds) {
  const auto compilationTarget = parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 9223372036854775806>, <id = 9223372036854775807>],
      connectivity = all_to_all, couplings = [],
      native_operations = explicit,
      operations = [<name = "cx",
          arity = #mqt.operation_arity<kind = fixed, value = 2>,
          num_parameters = 0,
          site_tuples = [<[9223372036854775806,
                           9223372036854775807]>]>]>)mlir");
  ASSERT_TRUE(compilationTarget);
  EXPECT_EQ(roundTrip(compilationTarget), compilationTarget);
}

TEST_F(MQTIRTest, RoundTripsSiteTupleCalibration) {
  const auto durationOnly = dyn_cast_if_present<mqt::SiteTupleAttr>(
      parseAttr(R"mlir(#mqt.site_tuple<[4, 7], duration = 0>)mlir"));
  ASSERT_TRUE(durationOnly);
  EXPECT_EQ(durationOnly.getDuration(), 0U);
  EXPECT_FALSE(durationOnly.getFidelity());
  EXPECT_EQ(roundTrip(durationOnly), durationOnly);

  const auto calibrated = dyn_cast_if_present<mqt::SiteTupleAttr>(parseAttr(
      R"mlir(#mqt.site_tuple<[4, 7], fidelity = 9.900000e-01 : f64, duration = 40>)mlir"));
  ASSERT_TRUE(calibrated);
  EXPECT_EQ(calibrated.getDuration(), 40U);
  EXPECT_EQ(calibrated.getFidelity().getValueAsDouble(), 0.99);
  EXPECT_EQ(roundTrip(calibrated), calibrated);
}

TEST_F(MQTIRTest, RepresentsUnrestrictedTargetFacts) {
  const auto unrestricted = dyn_cast_if_present<mqt::CompilationTargetAttr>(
      parseAttr(R"mlir(#mqt.compilation_target<
          sites = [<id = 0>], connectivity = all_to_all,
          couplings = [], native_operations = unrestricted, operations = []>)mlir"));
  ASSERT_TRUE(unrestricted);
  EXPECT_EQ(unrestricted.getConnectivity(), mqt::ConnectivityKind::AllToAll);
  EXPECT_EQ(unrestricted.getNativeOperations(),
            mqt::NativeOperationsKind::Unrestricted);
}

TEST_F(MQTIRTest, RejectsInvalidTargetLeaves) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.duration_unit<unit = "",
      scale_factor = 1.000000e+00 : f64>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.duration_unit<unit = "ns",
      scale_factor = 1.000000e+00 : f32>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.duration_unit<unit = "ns",
      scale_factor = 0.000000e+00 : f64>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site<id = -1>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site<id = 0, name = "">)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site<id = 0, t1 = 0>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site<id = 0, t1 =>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site<id = 0, t2 =>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.coupling<source = 0, target = 0>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site_tuple<[0, 0]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site_tuple<[-1]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site_tuple<[0], duration =>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site_tuple<[0],
      fidelity = 1.100000e+00 : f64>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.operation_arity<
      kind = variadic, value = 0>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "",
      arity = #mqt.operation_arity<kind = fixed, value = 1>,
      num_parameters = 0, site_tuples = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "gphase",
      arity = #mqt.operation_arity<kind = fixed, value = 0>,
      num_parameters = 1, site_tuples = [<[0]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "h",
      arity = #mqt.operation_arity<kind = variadic, value = 1>,
      num_parameters = 0, site_tuples = [<[0]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "cx",
      arity = #mqt.operation_arity<kind = fixed, value = 2>,
      num_parameters = 0, site_tuples = [<[0]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "x",
      arity = #mqt.operation_arity<kind = fixed, value = 1>,
      num_parameters = 0,
      site_tuples = [<[0]>, <[0]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "x",
      arity = #mqt.operation_arity<kind = fixed, value = 1>,
      num_parameters = 0, site_tuples = [], duration =>>)mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidCompilationTargets) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      name = "", sites = [<id = 0>], connectivity = all_to_all,
      couplings = [], native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [], connectivity = all_to_all, couplings = [],
      native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 0>], connectivity = all_to_all, couplings = [],
      native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 1>], connectivity = all_to_all,
      couplings = [<source = 0, target = 1>],
      native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = all_to_all, couplings = [],
      native_operations = unrestricted,
      operations = [<name = "x",
          arity = #mqt.operation_arity<kind = fixed, value = 1>,
          num_parameters = 0, site_tuples = []>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 1>], connectivity = explicit,
      couplings = [<source = 0, target = 2>],
      native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 1>], connectivity = explicit,
      couplings = [<source = 0, target = 1>, <source = 1, target = 0>],
      native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0, t1 = 100>], connectivity = all_to_all,
      couplings = [], native_operations = unrestricted, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = all_to_all, couplings = [],
      native_operations = explicit,
      operations = [<name = "x",
          arity = #mqt.operation_arity<kind = fixed, value = 1>,
          num_parameters = 0, site_tuples = [<[1]>]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = all_to_all, couplings = [],
      native_operations = explicit,
      operations = [<name = "cx",
          arity = #mqt.operation_arity<kind = fixed, value = 2>,
          num_parameters = 0, site_tuples = []>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = all_to_all, couplings = [],
      native_operations = explicit,
      operations = [<name = "h",
          arity = #mqt.operation_arity<kind = variadic, value = 2>,
          num_parameters = 0, site_tuples = []>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = all_to_all, couplings = [],
      native_operations = explicit,
      operations = [<name = "x",
          arity = #mqt.operation_arity<kind = fixed, value = 1>,
          num_parameters = 0, site_tuples = [], duration = 1>]>)mlir"));
}

TEST_F(MQTIRTest, ManagesAndFindsEntryPoint) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @helper() { return }
      func.func @main() attributes {mqt.entry_point} { return }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto main = mqt::getEntryPoint(*moduleOp);
  ASSERT_TRUE(main);
  EXPECT_EQ(main.getSymName(), "main");
  EXPECT_TRUE(mqt::isEntryPoint(main));

  mqt::removeEntryPoint(main);
  EXPECT_FALSE(mqt::isEntryPoint(main));
  EXPECT_FALSE(mqt::getEntryPoint(*moduleOp));

  auto helper = moduleOp->lookupSymbol<func::FuncOp>("helper");
  ASSERT_TRUE(helper);
  mqt::setEntryPoint(helper);
  EXPECT_TRUE(mqt::isEntryPoint(helper));
  EXPECT_EQ(mqt::getEntryPoint(*moduleOp), helper);
}

TEST_F(MQTIRTest, RejectsInvalidEntryPoints) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point = "yes"} { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func private @main() attributes {mqt.entry_point}
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @first() attributes {mqt.entry_point} { return }
      func.func @second() attributes {mqt.entry_point} { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {mqt.entry_point, value = 0 : i64}
            : () -> i64
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidInputNames) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @empty(%arg: f64 {mqt.input_name = ""}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @null(%arg: f64 {mqt.input_name = "a\00b"}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @wrong_type(%arg: f64 {mqt.input_name = 1 : i64}) { return }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsDuplicateInputNames) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main(%lhs: f64 {mqt.input_name = "theta"},
                      %rhs: i1 {mqt.input_name = "theta"}) {
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidInputGroups) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @incomplete(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "group"}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @empty_identity(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "", name = "theta",
                                 index = 0 : i64, size = 1 : i64}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @null_name(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "group", name = "theta\00",
                                 index = 0 : i64, size = 1 : i64}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @wrong_integer_type(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "group", name = "theta",
                                 index = 0 : i32, size = 1 : i64}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @wrong_name(%arg: f64 {mqt.input_name = "phi[0]",
          mqt.parameter_group = {identity = "group", name = "theta",
                                 index = 0 : i64, size = 1 : i64}}) { return }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsInputMetadataOnOperations) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {mqt.input_name = "theta", value = 0.0 : f64}
            : () -> f64
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {
          mqt.parameter_group = {identity = "group", name = "theta",
                                 index = 0 : i64, size = 1 : i64},
          value = 0.0 : f64} : () -> f64
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidRegisterNamesAndOwners) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @empty() {
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = ""}
            : !cbit.reg<2>
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %reg = memref.alloc() {mqt.register_name = "values"}
            : memref<2xf64>
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main(%arg: f64 {mqt.register_name = "q"}) {
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsDuplicateProgramNames) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %lhs = memref.alloc() {mqt.register_name = "state"}
            : memref<1x!qc.qubit>
        %rhs = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "state"}
            : !cbit.reg<2>
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main(%arg: f64 {mqt.input_name = "state"}) {
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "state"}
            : !cbit.reg<2>
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsUnknownMQTAttributes) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() attributes {mqt.unknown} { return }
    }
  )mlir"));
}
} // namespace
