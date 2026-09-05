/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_mqt_ir.cpp
/// Unit tests for the MQT metadata dialect.

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
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
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
    registry.insert<DLTIDialect, arith::ArithDialect, cbit::CBitDialect,
                    func::FuncDialect, memref::MemRefDialect, mqt::MQTDialect,
                    qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(const StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  [[nodiscard]] Attribute parseAttr(const StringRef source) const {
    return parseAttribute(source, context.get());
  }

  [[nodiscard]] OwningOpRef<ModuleOp> roundTrip(ModuleOp moduleOp) const {
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    moduleOp.print(stream);
    return parse(printed);
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

TEST_F(MQTIRTest, AcceptsSourceFunctionName) {
  EXPECT_TRUE(parse(R"mlir(
    module {
      func.func private @unique() attributes {mqt.source_name = "source"}
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidSourceFunctionNames) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func private @empty() attributes {mqt.source_name = ""}
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func private @null() attributes {mqt.source_name = "a\00b"}
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {mqt.source_name = "source", value = 0 : i64}
            : () -> i64
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

TEST_F(MQTIRTest, RejectsMutuallyRecursiveUnitaryFunctions) {
  for (StringRef source : {
           R"mlir(
             func.func private @first(%q: !qc.qubit) attributes {mqt.unitary} {
               qc.call @second(%q) : !qc.qubit
               return
             }
             func.func private @second(%q: !qc.qubit) attributes {mqt.unitary} {
               qc.call @first(%q) : !qc.qubit
               return
             }
           )mlir",
           R"mlir(
             func.func private @first(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %out = qco.call @second(%q) : (!qco.qubit) -> !qco.qubit
               return %out : !qco.qubit
             }
             func.func private @second(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %out = qco.call @first(%q) : (!qco.qubit) -> !qco.qubit
               return %out : !qco.qubit
             }
           )mlir",
       }) {
    SCOPED_TRACE(source.str());
    bool sawRecursion = false;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
      sawRecursion |= StringRef(diagnostic.str())
                          .contains("unitary function must not be recursive");
      return success();
    });
    EXPECT_FALSE(parse(source));
    EXPECT_TRUE(sawRecursion);
  }
}

TEST_F(MQTIRTest, RejectsEmptyUnitaryBodies) {
  for (StringRef source : {
           R"mlir(
             func.func private @empty(!qc.qubit) attributes {mqt.unitary} {
             ^bb0(%q: !qc.qubit):
             }
           )mlir",
           R"mlir(
             func.func private @empty(!qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
             ^bb0(%q: !qco.qubit):
             }
           )mlir",
       }) {
    SCOPED_TRACE(source.str());
    bool sawEmptyBody = false;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
      sawEmptyBody |= StringRef(diagnostic.str())
                          .contains("unitary function body must not be empty");
      return success();
    });
    EXPECT_FALSE(parse(source));
    EXPECT_TRUE(sawEmptyBody);
  }
}

TEST_F(MQTIRTest, RejectsCyclicUnitaryQubitFlow) {
  bool sawCycle = false;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
    sawCycle |= StringRef(diagnostic.str())
                    .contains("unitary QCO result has cyclic qubit flow");
    return success();
  });
  EXPECT_FALSE(parse(R"mlir(
    func.func private @cyclic(%q: !qco.qubit) -> !qco.qubit
        attributes {mqt.unitary} {
      %a = qco.h %b : !qco.qubit -> !qco.qubit
      %b = qco.h %a : !qco.qubit -> !qco.qubit
      return %b : !qco.qubit
    }
  )mlir"));
  EXPECT_TRUE(sawCycle);
}

TEST_F(MQTIRTest, RejectsMalformedUnitaryBodyOperations) {
  for (StringRef source : {
           R"mlir(
             func.func private @malformed(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %out = "qco.h"() : () -> !qco.qubit
               return %out : !qco.qubit
             }
           )mlir",
           R"mlir(
             func.func private @malformed(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %out = qco.inv (%arg = %q) {
                 %h = "qco.h"() : () -> !qco.qubit
                 qco.yield %h : !qco.qubit
               } : {!qco.qubit} -> {!qco.qubit}
               return %out : !qco.qubit
             }
           )mlir",
           R"mlir(
             func.func private @malformed(%q: !qc.qubit)
                 attributes {mqt.unitary} {
               %value = "memref.load"() : () -> f64
               return
             }
           )mlir",
           R"mlir(
             func.func private @malformed(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %value = "memref.load"() : () -> f64
               return %q : !qco.qubit
             }
           )mlir",
       }) {
    SCOPED_TRACE(source.str());
    bool sawOperandError = false;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
      sawOperandError |= StringRef(diagnostic.str()).contains("operand");
      return success();
    });
    EXPECT_FALSE(parse(source));
    EXPECT_TRUE(sawOperandError);
  }
}

TEST_F(MQTIRTest, RejectsMalformedCallsInUnitaryCallees) {
  for (StringRef source : {
           R"mlir(
             func.func private @first(%q: !qc.qubit) attributes {mqt.unitary} {
               qc.call @second(%q) : !qc.qubit
               return
             }
             func.func private @second(%q: !qc.qubit) attributes {mqt.unitary} {
               "qc.call"(%q) : (!qc.qubit) -> ()
               return
             }
           )mlir",
           R"mlir(
             func.func private @first(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %out = qco.call @second(%q) : (!qco.qubit) -> !qco.qubit
               return %out : !qco.qubit
             }
             func.func private @second(%q: !qco.qubit) -> !qco.qubit
                 attributes {mqt.unitary} {
               %out = "qco.call"(%q) : (!qco.qubit) -> !qco.qubit
               return %out : !qco.qubit
             }
           )mlir",
       }) {
    SCOPED_TRACE(source.str());
    bool sawMissingCallee = false;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
      sawMissingCallee |= StringRef(diagnostic.str()).contains("callee");
      return success();
    });
    EXPECT_FALSE(parse(source));
    EXPECT_TRUE(sawMissingCallee);
  }
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
TEST_F(MQTIRTest, RoundTripsTypedTargetEnvironment) {
  auto moduleOp = parse(R"mlir(
    module attributes {
      mqt.target_env = #mqt.target_env<
          compilation_target = #mqt.compilation_target<
              name = "device",
              sites = [<id = 10, name = "q0", t1 = 100, t2 = 80>,
                       <id = 20, name = "q1">],
              duration_unit = #mqt.duration_unit<unit = "ns",
                  scale_factor = 1.000000e-09 : f64>,
              connectivity = explicit,
              couplings = [<source = 10, target = 20>],
              native_operations = explicit,
              operations = [<name = "cx", arity = #mqt.operation_arity<kind = fixed, value = 2>,
                  num_parameters = 0,
                  site_tuples = [<[10, 20], duration = 50,
                      fidelity = 9.900000e-01 : f64>],
                  duration = 60, fidelity = 9.800000e-01 : f64>]>,
          payload_specification = #mqt.payload_spec<
              format = #mqt.payload_format<id = "vendor-ir",
                  version = "4.2.0", profile = "dynamic", encoding = binary>,
              capabilities = [<id = "integer-computation", value = 64,
                  constraints = [<id = "max-control-flow-depth", value = 8>]>],
              optional_capabilities_known = false>,
          extensions = #dlti.map<"vendor.queue_depth" = 8 : i64>>
    } {
      func.func @main() { return }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  const auto targetEnv =
      (*moduleOp)->getAttrOfType<mqt::TargetEnvAttr>(mqt::TargetEnvAttr::name);
  ASSERT_TRUE(targetEnv);
  const auto compilationTarget = targetEnv.getCompilationTarget();
  EXPECT_EQ(compilationTarget.getName().getValue(), "device");
  ASSERT_EQ(compilationTarget.getSites().size(), 2U);
  EXPECT_EQ(compilationTarget.getSites()[0].getId(), 10);
  EXPECT_EQ(compilationTarget.getSites()[1].getId(), 20);
  EXPECT_EQ(compilationTarget.getConnectivity(),
            mqt::ConnectivityKind::Explicit);
  EXPECT_EQ(compilationTarget.getNativeOperations(),
            mqt::NativeOperationsKind::Explicit);
  ASSERT_EQ(compilationTarget.getOperations().size(), 1U);
  const auto operationSites = compilationTarget.getOperations()
                                  .front()
                                  .getSiteTuples()
                                  .front()
                                  .getSites();
  ASSERT_EQ(operationSites.size(), 2U);
  EXPECT_EQ(operationSites[0], 10);
  EXPECT_EQ(operationSites[1], 20);

  const auto payloadSpecification = targetEnv.getPayloadSpecification();
  EXPECT_EQ(payloadSpecification.getFormat().getId().getValue(), "vendor-ir");
  EXPECT_EQ(payloadSpecification.getFormat().getVersion().getValue(), "4.2.0");
  EXPECT_EQ(payloadSpecification.getFormat().getProfile().getValue(),
            "dynamic");
  EXPECT_EQ(payloadSpecification.getFormat().getEncoding(),
            mqt::PayloadEncoding::Binary);
  EXPECT_FALSE(payloadSpecification.getOptionalCapabilitiesKnown());
  ASSERT_EQ(payloadSpecification.getCapabilities().size(), 1U);
  ASSERT_EQ(
      payloadSpecification.getCapabilities().front().getConstraints().size(),
      1U);

  auto query = cast<DLTIQueryInterface>(targetEnv);
  auto targetResult = query.query(StringAttr::get(
      context.get(), mqt::TargetEnvAttr::kCompilationTargetKey));
  ASSERT_TRUE(succeeded(targetResult));
  EXPECT_EQ(*targetResult, compilationTarget);
  auto payloadResult = query.query(StringAttr::get(
      context.get(), mqt::TargetEnvAttr::kPayloadSpecificationKey));
  ASSERT_TRUE(succeeded(payloadResult));
  EXPECT_EQ(*payloadResult, payloadSpecification);
  auto extensionResult =
      query.query(StringAttr::get(context.get(), "vendor.queue_depth"));
  ASSERT_TRUE(succeeded(extensionResult));
  EXPECT_EQ(cast<IntegerAttr>(*extensionResult).getInt(), 8);
  EXPECT_TRUE(failed(query.query(IntegerType::get(context.get(), 32))));

  const auto reparsed = roundTrip(*moduleOp);
  ASSERT_TRUE(reparsed);
  EXPECT_EQ((*reparsed)->getAttr(mqt::TargetEnvAttr::name), targetEnv);
}

TEST_F(MQTIRTest, RepresentsEmptyPayloadCapabilities) {
  const auto payload = dyn_cast_if_present<mqt::PayloadSpecAttr>(parseAttr(
      R"mlir(#mqt.payload_spec<format = #mqt.payload_format<
          id = "openqasm", version = "3.0.0", profile = "", encoding = text>,
          capabilities = [], optional_capabilities_known = true>)mlir"));
  ASSERT_TRUE(payload);
  EXPECT_TRUE(payload.getCapabilities().empty());
  EXPECT_TRUE(payload.getOptionalCapabilitiesKnown());
}

TEST_F(MQTIRTest, RejectsInvalidPayloadContracts) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_format<id = "",
      version = "2.1.0", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_format<id = "qir",
      version = "2.1.0", profile = "base\00suffix", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_format<id = "qir",
      version = "2.1", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_format<id = "qir",
      version = "02.1.0", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_format<id = "qir",
      version = "2.1.0-beta", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(
      parseAttr(R"mlir(#mqt.program_constraint<id = "", value = 1>)mlir"));
  EXPECT_FALSE(parseAttr(
      R"mlir(#mqt.program_constraint<id = "bad\00id", value = 1>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.program_capability<id = "", value = 1,
      constraints = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.program_capability<id = "bad\00id",
      value = 1, constraints = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.program_capability<id = "loops", value = 1,
      constraints = [<id = "max-depth", value = 4>,
                     <id = "max-depth", value = 8>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_spec<
      format = #mqt.payload_format<id = "qir", version = "2.1.0",
          profile = "base", encoding = text>,
      capabilities = [<id = "loops", value = 1, constraints = []>,
                      <id = "loops", value = 1, constraints = []>],
      optional_capabilities_known = true>)mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidTargetEnvironmentExtensions) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.target_env<
      compilation_target = #mqt.compilation_target<
          sites = [<id = 0>], connectivity = all_to_all, couplings = [],
          native_operations = unrestricted, operations = []>,
      payload_specification = #mqt.payload_spec<
          format = #mqt.payload_format<id = "qir", version = "2.1.0",
              profile = "base", encoding = binary>, capabilities = [],
          optional_capabilities_known = false>,
      extensions = #dlti.map<"unnamespaced" = 1 : i64>>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.target_env<
      compilation_target = #mqt.compilation_target<
          sites = [<id = 0>], connectivity = all_to_all, couplings = [],
          native_operations = unrestricted, operations = []>,
      payload_specification = #mqt.payload_spec<
          format = #mqt.payload_format<id = "qir", version = "2.1.0",
              profile = "base", encoding = binary>, capabilities = [],
          optional_capabilities_known = false>,
      extensions = #dlti.map<"mqt.payload_specification" = 1 : i64>>)mlir"));
}

TEST_F(MQTIRTest, RejectsTargetEnvironmentOutsideModule) {
  EXPECT_FALSE(parse(R"mlir(
    module attributes {mqt.target_env = "invalid"} {}
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() attributes {
        mqt.target_env = #mqt.target_env<
            compilation_target = #mqt.compilation_target<
                sites = [<id = 0>], connectivity = all_to_all, couplings = [],
                native_operations = unrestricted, operations = []>,
            payload_specification = #mqt.payload_spec<
                format = #mqt.payload_format<id = "qir",
                    version = "2.1.0", profile = "base", encoding = binary>,
                capabilities = [], optional_capabilities_known = false>>
      } { return }
    }
  )mlir"));
}

} // namespace
