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
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

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
                    DLTIDialect, memref::MemRefDialect, mqt::MQTDialect,
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
};

TEST_F(MQTIRTest, AcceptsProgramInputAndRegisterNames) {
  EXPECT_TRUE(parse(R"mlir(
    module {
      func.func @qc(%theta: f64 {mqt.input_name = "theta"}) {
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
              operations = [<name = "cx", arity = 2,
                  num_parameters = 0,
                  site_tuples = [<sites = [10, 20], duration = 50,
                      fidelity = 9.900000e-01 : f64>],
                  duration = 60, fidelity = 9.800000e-01 : f64>]>,
          payload_env = #mqt.payload_env<
              descriptor = #mqt.payload_descriptor<id = "vendor-ir",
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

  const auto payloadEnv = targetEnv.getPayloadEnv();
  EXPECT_EQ(payloadEnv.getDescriptor().getId().getValue(), "vendor-ir");
  EXPECT_EQ(payloadEnv.getDescriptor().getVersion().getValue(), "4.2.0");
  EXPECT_EQ(payloadEnv.getDescriptor().getProfile().getValue(), "dynamic");
  EXPECT_EQ(payloadEnv.getDescriptor().getEncoding(),
            mqt::PayloadEncoding::Binary);
  EXPECT_FALSE(payloadEnv.getOptionalCapabilitiesKnown());
  ASSERT_EQ(payloadEnv.getCapabilities().size(), 1U);
  ASSERT_EQ(payloadEnv.getCapabilities().front().getConstraints().size(), 1U);

  auto query = cast<DLTIQueryInterface>(targetEnv);
  auto targetResult = query.query(StringAttr::get(
      context.get(), mqt::TargetEnvAttr::kCompilationTargetKey));
  ASSERT_TRUE(succeeded(targetResult));
  EXPECT_EQ(*targetResult, compilationTarget);
  auto payloadResult = query.query(
      StringAttr::get(context.get(), mqt::TargetEnvAttr::kPayloadEnvKey));
  ASSERT_TRUE(succeeded(payloadResult));
  EXPECT_EQ(*payloadResult, payloadEnv);
  auto extensionResult =
      query.query(StringAttr::get(context.get(), "vendor.queue_depth"));
  ASSERT_TRUE(succeeded(extensionResult));
  EXPECT_EQ(cast<IntegerAttr>(*extensionResult).getInt(), 8);
  EXPECT_TRUE(failed(query.query(IntegerType::get(context.get(), 32))));

  const auto reparsed = roundTrip(*moduleOp);
  ASSERT_TRUE(reparsed);
  EXPECT_EQ((*reparsed)->getAttr(mqt::TargetEnvAttr::name), targetEnv);
}

TEST_F(MQTIRTest, RepresentsUnknownAndUnrestrictedTargetFacts) {
  const auto unknown = dyn_cast_if_present<mqt::CompilationTargetAttr>(
      parseAttr(R"mlir(#mqt.compilation_target<
          sites = [<id = 0>], connectivity = unknown, couplings = [],
          native_operations = unknown, operations = []>)mlir"));
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown.getConnectivity(), mqt::ConnectivityKind::Unknown);
  EXPECT_EQ(unknown.getNativeOperations(), mqt::NativeOperationsKind::Unknown);

  const auto unrestricted = dyn_cast_if_present<mqt::CompilationTargetAttr>(
      parseAttr(R"mlir(#mqt.compilation_target<
          sites = [<id = 0>], connectivity = all_to_all,
          couplings = [], native_operations = unrestricted, operations = []>)mlir"));
  ASSERT_TRUE(unrestricted);
  EXPECT_EQ(unrestricted.getConnectivity(), mqt::ConnectivityKind::AllToAll);
  EXPECT_EQ(unrestricted.getNativeOperations(),
            mqt::NativeOperationsKind::Unrestricted);

  const auto payload = dyn_cast_if_present<mqt::PayloadEnvAttr>(parseAttr(
      R"mlir(#mqt.payload_env<descriptor = #mqt.payload_descriptor<
          id = "openqasm", version = "3.0.0", profile = "", encoding = text>,
          capabilities = [], optional_capabilities_known = true>)mlir"));
  ASSERT_TRUE(payload);
  EXPECT_TRUE(payload.getCapabilities().empty());
  EXPECT_TRUE(payload.getOptionalCapabilitiesKnown());
}

TEST_F(MQTIRTest, RejectsInvalidPayloadContracts) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_descriptor<id = "",
      version = "2.1.0", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_descriptor<id = "qir",
      version = "2.1.0", profile = "base\00suffix", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_descriptor<id = "qir",
      version = "2.1", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_descriptor<id = "qir",
      version = "02.1.0", profile = "base", encoding = text>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_descriptor<id = "qir",
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
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.payload_env<
      descriptor = #mqt.payload_descriptor<id = "qir", version = "2.1.0",
          profile = "base", encoding = text>,
      capabilities = [<id = "loops", value = 1, constraints = []>,
                      <id = "loops", value = 1, constraints = []>],
      optional_capabilities_known = true>)mlir"));
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
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.coupling<source = 0, target = 0>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site_tuple<sites = [0, 0]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.site_tuple<sites = [0],
      fidelity = 1.100000e+00 : f64>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "", arity = 1,
      num_parameters = 0, site_tuples = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "x", arity = 0,
      num_parameters = 0, site_tuples = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "cx", arity = 2,
      num_parameters = 0, site_tuples = [<sites = [0]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.native_operation<name = "x", arity = 1,
      num_parameters = 0,
      site_tuples = [<sites = [0]>, <sites = [0]>]>)mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidCompilationTargets) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      name = "", sites = [<id = 0>], connectivity = unknown,
      couplings = [], native_operations = unknown, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [], connectivity = unknown, couplings = [],
      native_operations = unknown, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 0>], connectivity = unknown, couplings = [],
      native_operations = unknown, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 1>], connectivity = unknown,
      couplings = [<source = 0, target = 1>], native_operations = unknown,
      operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = unknown, couplings = [],
      native_operations = unrestricted,
      operations = [<name = "x", arity = 1, num_parameters = 0,
          site_tuples = []>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 1>], connectivity = explicit,
      couplings = [<source = 0, target = 2>], native_operations = unknown,
      operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>, <id = 1>], connectivity = explicit,
      couplings = [<source = 0, target = 1>, <source = 1, target = 0>],
      native_operations = unknown, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0, t1 = 100>], connectivity = unknown,
      couplings = [], native_operations = unknown, operations = []>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = unknown, couplings = [],
      native_operations = explicit,
      operations = [<name = "x", arity = 1, num_parameters = 0,
          site_tuples = [<sites = [1]>]>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = unknown, couplings = [],
      native_operations = explicit,
      operations = [<name = "cx", arity = 2, num_parameters = 0,
          site_tuples = []>]>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.compilation_target<
      sites = [<id = 0>], connectivity = unknown, couplings = [],
      native_operations = explicit,
      operations = [<name = "x", arity = 1, num_parameters = 0,
          site_tuples = [], duration = 1>]>)mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidTargetEnvironmentExtensions) {
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.target_env<
      compilation_target = #mqt.compilation_target<
          sites = [<id = 0>], connectivity = unknown, couplings = [],
          native_operations = unknown, operations = []>,
      payload_env = #mqt.payload_env<
          descriptor = #mqt.payload_descriptor<id = "qir", version = "2.1.0",
              profile = "base", encoding = binary>, capabilities = [],
          optional_capabilities_known = false>,
      extensions = #dlti.map<"unnamespaced" = 1 : i64>>)mlir"));
  EXPECT_FALSE(parseAttr(R"mlir(#mqt.target_env<
      compilation_target = #mqt.compilation_target<
          sites = [<id = 0>], connectivity = unknown, couplings = [],
          native_operations = unknown, operations = []>,
      payload_env = #mqt.payload_env<
          descriptor = #mqt.payload_descriptor<id = "qir", version = "2.1.0",
              profile = "base", encoding = binary>, capabilities = [],
          optional_capabilities_known = false>,
      extensions = #dlti.map<"mqt.payload_env" = 1 : i64>>)mlir"));
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
                sites = [<id = 0>], connectivity = unknown, couplings = [],
                native_operations = unknown, operations = []>,
            payload_env = #mqt.payload_env<
                descriptor = #mqt.payload_descriptor<id = "qir",
                    version = "2.1.0", profile = "base", encoding = binary>,
                capabilities = [], optional_capabilities_known = false>>
      } { return }
    }
  )mlir"));
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

TEST_F(MQTIRTest, RejectsInputNameOnOperation) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {mqt.input_name = "theta", value = 0.0 : f64}
            : () -> f64
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
