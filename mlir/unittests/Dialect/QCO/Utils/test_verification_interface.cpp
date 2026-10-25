/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QCO/Utils/Verification.h"
#include "mqt/Dialect/QCO/Utils/VerificationInterface.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

namespace {

class VerificationInterfaceTest : public ::testing::Test {
protected:
  MLIRContext context;
  ScopedDiagnosticHandler diagnostics{&context,
                                      [](Diagnostic&) { return success(); }};

  VerificationInterfaceTest() {
    context
        .loadDialect<arith::ArithDialect, func::FuncDialect, mqt::MQTDialect,
                     QCODialect, qtensor::QTensorDialect, cbit::CBitDialect>();
  }

  OwningOpRef<ModuleOp> parse(llvm::StringRef source) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  static std::string fixture(llvm::StringRef name) {
    const auto path = std::filesystem::path(__FILE__).parent_path() /
                      "testdata" / "verification" / name.str();
    std::ifstream stream(path);
    EXPECT_TRUE(stream.good()) << path;
    return {std::istreambuf_iterator<char>{stream},
            std::istreambuf_iterator<char>{}};
  }

  static std::string replace(std::string source, llvm::StringRef oldText,
                             llvm::StringRef newText) {
    const auto position = source.find(oldText.str());
    EXPECT_NE(position, std::string::npos) << oldText.str();
    if (position != std::string::npos) {
      source.replace(position, oldText.size(), newText.str());
    }
    return source;
  }

  static func::FuncOp entry(ModuleOp moduleOp) {
    return moduleOp.lookupSymbol<func::FuncOp>("main");
  }

  static DDResolvedExecution
  staticTrace(std::initializer_list<uint64_t> sites) {
    DDResolvedExecution trace;
    trace.status = DDTraceStatus::Complete;
    for (const auto site : sites) {
      trace.roots.push_back({
          .kind = DDWireRoot::Kind::Static,
          .wire = static_cast<dd::Qubit>(trace.numQubits++),
          .staticId = site,
      });
    }
    return trace;
  }
};

constexpr llvm::StringLiteral SPARSE_LAYOUT = R"mlir(
module attributes {mqt.layout = {
  initial = array<i64: 1, 0, 2, 3>, input_count = 3 : i64,
  routing = array<i64: 1, 2, 0, 3>, sites = array<i64: 10, 30, 20, 40>}} {
  func.func @main() {
    %a = qco.static 30 : !qco.qubit
    %b = qco.static 20 : !qco.qubit
    %c = qco.static 10 : !qco.qubit
    qco.sink %a : !qco.qubit
    qco.sink %b : !qco.qubit
    qco.sink %c : !qco.qubit
    return
  }
})mlir";

TEST_F(VerificationInterfaceTest,
       ComposesSparseLayoutAndPreservesIdleWorkspace) {
  auto moduleOp = parse(SPARSE_LAYOUT);
  ASSERT_TRUE(moduleOp);
  auto result = resolveVerificationInterface(entry(*moduleOp),
                                             staticTrace({30, 20, 10}), {}, 3);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->numInputs, 3);
  EXPECT_EQ(result->numQubits, 4);
  EXPECT_EQ(result->inputWires, (std::vector<dd::Qubit>{0, 2, 1, 3}));
  EXPECT_EQ(result->outputWires, (std::vector<dd::Qubit>{1, 0, 2, 3}));
  EXPECT_EQ(result->zeroInputs, (std::vector<bool>{false, false, false, true}));
  EXPECT_EQ(result->cleanOutputs, (std::vector<dd::Qubit>{3}));
}

TEST_F(VerificationInterfaceTest, InputCountCannotEraseAReferencePort) {
  auto source = SPARSE_LAYOUT.str();
  source.replace(source.find("input_count = 3"), 15, "input_count = 2");
  auto moduleOp = parse(source);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(failed(resolveVerificationInterface(
      entry(*moduleOp), staticTrace({30, 20, 10}), {}, 3)));
  VerificationContract contract;
  contract.numInputs = 3;
  EXPECT_TRUE(failed(resolveVerificationInterface(
      entry(*moduleOp), staticTrace({30, 20, 10}), contract)));
}

TEST_F(VerificationInterfaceTest, QuantumReturnsMustAgreeWithLayoutOutputs) {
  auto source = replace(SPARSE_LAYOUT.str(), "@main()", R"mlir(@main()
      -> (!qco.qubit, !qco.qubit, !qco.qubit))mlir");
  for (const auto* name : {"%a", "%b", "%c"}) {
    source = replace(std::move(source),
                     std::string("qco.sink ") + name + " : !qco.qubit", "");
  }
  source = replace(std::move(source), "return",
                   "return %b, %a, %c : !qco.qubit, !qco.qubit, !qco.qubit");
  auto moduleOp = parse(source);
  ASSERT_TRUE(moduleOp);
  auto trace = staticTrace({30, 20, 10});
  trace.quantumReturns = {1, 0, 2};
  const auto result = resolveVerificationInterface(entry(*moduleOp), trace);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->outputWires, (std::vector<dd::Qubit>{1, 0, 2, 3}));
  trace.quantumReturns = {0, 1, 2};
  EXPECT_TRUE(failed(resolveVerificationInterface(entry(*moduleOp), trace)));
}

TEST_F(VerificationInterfaceTest, RejectsLayoutAndRootIdentityCorruption) {
  auto moduleOp = parse(SPARSE_LAYOUT);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(failed(resolveVerificationInterface(
      entry(*moduleOp), staticTrace({30, 20, 99}), {}, 3)));
  EXPECT_TRUE(failed(resolveVerificationInterface(
      entry(*moduleOp), staticTrace({30, 20, 20}), {}, 3)));
  auto malformed = SPARSE_LAYOUT.str();
  const std::string initial = "initial = array<i64: 1, 0, 2, 3>";
  malformed.replace(malformed.find(initial), initial.size(),
                    "initial = array<i64: 1, 1, 2, 3>");
  auto invalid = parse(malformed);
  if (invalid) {
    EXPECT_TRUE(failed(resolveVerificationInterface(
        entry(*invalid), staticTrace({30, 20, 10}), {}, 3)));
  }
}

TEST_F(VerificationInterfaceTest, SourceTagsSurviveShrinkingAndCircuitElision) {
  auto moduleOp = parse(R"mlir(
module attributes {mqt.source_qubit_count = 4 : i64, mqt.layout = {
  initial = array<i64: 0, 1, 2>, input_count = 3 : i64,
  routing = array<i64: 1, 2, 0>}} {
  func.func @main() {
    %n = arith.constant 3 : index
    %q = qtensor.alloc(%n) {mqt.source_qubit_indices = array<i64: 0, 2, 3>}
      : tensor<3x!qco.qubit>
    qtensor.dealloc %q : tensor<3x!qco.qubit>
    return
  }
})mlir");
  ASSERT_TRUE(moduleOp);
  DDResolvedExecution trace;
  trace.status = DDTraceStatus::Complete;
  trace.numQubits = 3;
  for (size_t index = 0; index < 3; ++index) {
    trace.roots.push_back({
        .kind = DDWireRoot::Kind::Allocation,
        .wire = static_cast<dd::Qubit>(index),
        .elementIndex = index,
        .allocationIndex = 0,
    });
  }
  auto result = resolveVerificationInterface(entry(*moduleOp), trace, {}, 4);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->inputWires, (std::vector<dd::Qubit>{0, 3, 1, 2}));
  EXPECT_EQ(result->outputWires, (std::vector<dd::Qubit>{1, 3, 2, 0}));
  EXPECT_EQ(result->zeroInputs, std::vector<bool>(4, false));
}

TEST_F(VerificationInterfaceTest,
       StaticRootsRetainNumericIdentityWithoutLayout) {
  auto moduleOp = parse(R"mlir(module {
    func.func @main() {
      %q = qco.static 3 : !qco.qubit
      qco.sink %q : !qco.qubit
      return
    }
  })mlir");
  ASSERT_TRUE(moduleOp);
  auto result =
      resolveVerificationInterface(entry(*moduleOp), staticTrace({3}));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->numInputs, 4);
  EXPECT_EQ(result->inputWires, (std::vector<dd::Qubit>{1, 2, 3, 0}));
  EXPECT_TRUE(failed(
      resolveVerificationInterface(entry(*moduleOp), staticTrace({4}), {}, 4)));
}

TEST_F(VerificationInterfaceTest,
       InitializedAllocationsDoNotInitializeArguments) {
  auto moduleOp = parse(R"mlir(module {
    func.func @main(%arg: !qco.qubit) -> !qco.qubit {
      %q = qco.alloc : !qco.qubit
      %n = arith.constant 2 : index
      %t = qtensor.alloc(%n) : tensor<2x!qco.qubit>
      qco.sink %q : !qco.qubit
      qtensor.dealloc %t : tensor<2x!qco.qubit>
      return %arg : !qco.qubit
    }
  })mlir");
  ASSERT_TRUE(moduleOp);
  DDResolvedExecution trace;
  trace.status = DDTraceStatus::Complete;
  trace.numQubits = 4;
  trace.roots = {
      {.kind = DDWireRoot::Kind::Argument, .wire = 0},
      {.kind = DDWireRoot::Kind::Allocation, .wire = 1},
      {.kind = DDWireRoot::Kind::Allocation, .wire = 2, .allocationIndex = 1},
      {
          .kind = DDWireRoot::Kind::Allocation,
          .wire = 3,
          .elementIndex = 1,
          .allocationIndex = 1,
      },
  };
  trace.quantumReturns = {0};
  VerificationContract contract;
  contract.domain = VerificationDomain::InitializedProgram;
  auto result = resolveVerificationInterface(entry(*moduleOp), trace, contract);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->numInputs, 4);
  EXPECT_EQ(result->zeroInputs, (std::vector<bool>{false, true, true, true}));
  EXPECT_EQ(result->cleanOutputs, (std::vector<dd::Qubit>{1, 2, 3}));
  EXPECT_TRUE(failed(resolveVerificationInterface(entry(*moduleOp), trace)));
  contract.domain = VerificationDomain::CircuitInputs;
  contract.zeroAncillas = {1, 2, 3};
  EXPECT_TRUE(succeeded(
      resolveVerificationInterface(entry(*moduleOp), trace, contract)));
}

TEST_F(VerificationInterfaceTest, ExplicitAncillasMustBeDistinctSourceSlots) {
  auto moduleOp = parse(SPARSE_LAYOUT);
  ASSERT_TRUE(moduleOp);
  VerificationContract contract;
  contract.zeroAncillas = {1};
  auto result = resolveVerificationInterface(
      entry(*moduleOp), staticTrace({30, 20, 10}), contract);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->zeroInputs, (std::vector<bool>{false, true, false, true}));
  for (const auto& invalid :
       {std::vector<size_t>{1, 1}, std::vector<size_t>{3}}) {
    contract.zeroAncillas = invalid;
    EXPECT_TRUE(failed(resolveVerificationInterface(
        entry(*moduleOp), staticTrace({30, 20, 10}), contract)));
  }
}

TEST_F(VerificationInterfaceTest, ReadoutRetainsSparseDestinationsAndNames) {
  auto moduleOp = parse(SPARSE_LAYOUT);
  ASSERT_TRUE(moduleOp);
  auto trace = staticTrace({30, 20, 10});
  trace.outputs = {
      {.resultIndex = 0, .bitIndex = 7, .name = "c", .wire = 0},
      {.resultIndex = 0, .bitIndex = 2, .name = "c", .wire = 2},
      {.resultIndex = 0, .bitIndex = 5, .name = "c", .wire = 1},
      {.resultIndex = 0, .bitIndex = 0, .name = "c", .constant = false},
  };
  auto result = resolveVerificationInterface(entry(*moduleOp), trace);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->outputLabels,
            (std::vector<std::string>{"result0:c[0]", "result0:c[2]",
                                      "result0:c[5]", "result0:c[7]"}));
  EXPECT_EQ(result->outputs[0].constant, false);
  EXPECT_EQ(result->outputs[1].wire, 2);
  trace.outputs[0].name = "other";
  auto mutated = resolveVerificationInterface(entry(*moduleOp), trace);
  ASSERT_TRUE(succeeded(mutated));
  EXPECT_NE(mutated->outputLabels, result->outputLabels);
  trace.outputs[0].name = "c";
  VerificationContract contract;
  contract.observedOutputs = {"result0:c[7]", "result0:c[2]"};
  auto selected =
      resolveVerificationInterface(entry(*moduleOp), trace, contract);
  ASSERT_TRUE(succeeded(selected));
  EXPECT_EQ(selected->outputLabels, contract.observedOutputs);
  EXPECT_EQ(selected->outputs[0].wire, 0);
  contract.observedOutputs = {"result0:c[7]", "result0:c[7]"};
  EXPECT_TRUE(
      failed(resolveVerificationInterface(entry(*moduleOp), trace, contract)));
  contract.observedOutputs = {"result0:c[1]"};
  EXPECT_TRUE(
      failed(resolveVerificationInterface(entry(*moduleOp), trace, contract)));
}

TEST_F(VerificationInterfaceTest, UndefinedAndAliasedReadoutDestinationsFail) {
  auto moduleOp = parse(SPARSE_LAYOUT);
  ASSERT_TRUE(moduleOp);
  auto trace = staticTrace({30, 20, 10});
  trace.outputs = {{.resultIndex = 0, .bitIndex = 2, .name = "c"}};
  EXPECT_TRUE(failed(resolveVerificationInterface(entry(*moduleOp), trace)));
  trace.outputs[0].wire = 0;
  trace.outputs.push_back(trace.outputs[0]);
  trace.outputs[1].name = "other";
  EXPECT_TRUE(failed(resolveVerificationInterface(entry(*moduleOp), trace)));
}

TEST_F(VerificationInterfaceTest, RealCompilationPreservesAsymmetricCircuit) {
  auto source = parse(fixture("before-elision.mlir"));
  ASSERT_TRUE(source);
  for (const auto* filename : {"after-elision.mlir", "sparse-sites.mlir"}) {
    SCOPED_TRACE(filename);
    auto compiled = parse(fixture(filename));
    ASSERT_TRUE(compiled);
    for (const auto method :
         {VerificationMethod::Construction, VerificationMethod::Alternating}) {
      VerificationOptions options;
      options.method = method;
      const auto result =
          qco::verify(entry(*source), entry(*compiled), {},
                      DDArgumentBindings(), DDArgumentBindings(), options);
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(result->verdict, VerificationVerdict::Equivalent)
          << result->reason;
      EXPECT_EQ(result->numInputs, 3);
      EXPECT_EQ(result->numFreeInputs, 3);
    }
  }
}

TEST_F(VerificationInterfaceTest,
       CapabilityTargetKeepsExplicitArgumentIdentity) {
  const auto compiled = fixture("sparse-sites.mlir");
  const auto targetStart = compiled.find("mqt.target_env =");
  const auto functionStart = compiled.find("  func.func");
  ASSERT_NE(targetStart, std::string::npos);
  ASSERT_NE(functionStart, std::string::npos);
  const auto prefix = "module attributes {" +
                      compiled.substr(targetStart, functionStart - targetStart);
  auto moduleOp = parse(prefix + R"mlir(
    func.func @main(%q: !qco.qubit) -> !qco.qubit {
      return %q : !qco.qubit
    }
  })mlir");
  ASSERT_TRUE(moduleOp);
  DDResolvedExecution trace;
  trace.status = DDTraceStatus::Complete;
  trace.numQubits = 1;
  trace.roots = {{.kind = DDWireRoot::Kind::Argument, .wire = 0}};
  trace.quantumReturns = {0};
  const auto result = resolveVerificationInterface(entry(*moduleOp), trace);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->inputWires, (std::vector<dd::Qubit>{0}));
  EXPECT_EQ(result->zeroInputs, (std::vector<bool>{false}));
  auto placed = parse(prefix + compiled.substr(functionStart));
  ASSERT_TRUE(placed);
  EXPECT_TRUE(failed(
      resolveVerificationInterface(entry(*placed), staticTrace({10, 30, 20}))));
}

TEST_F(VerificationInterfaceTest, CompilationFaultsCannotNormalizeAway) {
  auto source = parse(fixture("before-elision.mlir"));
  ASSERT_TRUE(source);
  const auto compiled = fixture("sparse-sites.mlir");
  const std::vector<std::pair<std::string, std::string>> mutations{
      {"2.050000e-01", "2.150000e-01"},
      {"qco.x %q_44", "qco.z %q_44"},
      {
          "%controls_out, %targets_out = qco.ctrl(%q_24) targets (%q_44 = "
          "%q_26)",
          "%targets_out, %controls_out = qco.ctrl(%q_26) targets (%q_44 = "
          "%q_24)",
      },
      {"2.8894042348957698", "2.9894042348957698"},
      {"initial = array<i64: 1, 0, 2>", "initial = array<i64: 0, 1, 2>"},
      {"routing = array<i64: 1, 2, 0>", "routing = array<i64: 2, 1, 0>"},
      {"sites = array<i64: 10, 30, 20>", "sites = array<i64: 30, 10, 20>"},
  };
  for (const auto& [original, mutation] : mutations) {
    SCOPED_TRACE(original);
    auto mutated = parse(replace(compiled, original, mutation));
    ASSERT_TRUE(mutated);
    VerificationOptions options;
    options.method = VerificationMethod::Construction;
    const auto result =
        qco::verify(entry(*source), entry(*mutated), {}, DDArgumentBindings(),
                    DDArgumentBindings(), options);
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->verdict, VerificationVerdict::NotEquivalent)
        << result->reason;
  }
  auto erased = parse(replace(compiled, "input_count = 3", "input_count = 2"));
  ASSERT_TRUE(erased);
  const auto result = qco::verify(entry(*source), entry(*erased));
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, VerificationVerdict::Unsupported);
}

TEST_F(VerificationInterfaceTest, PrivateWorkspaceMustReturnToZero) {
  auto source = parse(fixture("before-elision.mlir"));
  auto compiled = parse(fixture("workspace.mlir"));
  ASSERT_TRUE(source);
  ASSERT_TRUE(compiled);
  VerificationOptions options;
  options.method = VerificationMethod::Construction;
  auto result =
      qco::verify(entry(*source), entry(*compiled), {}, DDArgumentBindings(),
                  DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, VerificationVerdict::Equivalent) << result->reason;
  EXPECT_EQ(result->numQubits, 4);
  auto dirty = parse(replace(fixture("workspace.mlir"), "    return", R"mlir(
    %workspace = qco.static 3 : !qco.qubit
    %dirty = qco.x %workspace : !qco.qubit
    qco.sink %dirty : !qco.qubit
    return)mlir"));
  ASSERT_TRUE(dirty);
  result = qco::verify(entry(*source), entry(*dirty), {}, DDArgumentBindings(),
                       DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, VerificationVerdict::Unsupported)
      << result->reason;
  options.method = VerificationMethod::Auto;
  result = qco::verify(entry(*source), entry(*dirty), {}, DDArgumentBindings(),
                       DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, VerificationVerdict::Unsupported)
      << result->reason;
  options.method = VerificationMethod::Simulation;
  result = qco::verify(entry(*source), entry(*compiled), {},
                       DDArgumentBindings(), DDArgumentBindings(), options);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->verdict, VerificationVerdict::Unsupported);
}

TEST_F(VerificationInterfaceTest, SparseReadoutMatchesSourceDestinations) {
  auto sourceText = replace(fixture("before-elision.mlir"), "@main()",
                            "@main() -> !cbit.reg<9>");
  sourceText =
      replace(std::move(sourceText),
              "    qtensor.dealloc %qs_21 : tensor<3x!qco.qubit>\n    return",
              R"mlir(
    %zero = arith.constant 0 : i9
    %five = arith.constant 5 : index
    %seven = arith.constant 7 : index
    %c = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "c"} : !cbit.reg<9>
    cbit.write %zero, %c : i9, !cbit.reg<9>
    %rest0, %q0 = qtensor.extract %qs_21[%c0] : tensor<3x!qco.qubit>
    %m0, %b0 = qco.measure %q0 : !qco.qubit
    %back0 = qtensor.insert %m0 into %rest0[%c0] : tensor<3x!qco.qubit>
    %rest1, %q1 = qtensor.extract %back0[%c1] : tensor<3x!qco.qubit>
    %m1, %b1 = qco.measure %q1 : !qco.qubit
    %back1 = qtensor.insert %m1 into %rest1[%c1] : tensor<3x!qco.qubit>
    %rest2, %q2 = qtensor.extract %back1[%c2] : tensor<3x!qco.qubit>
    %m2, %b2 = qco.measure %q2 : !qco.qubit
    %back2 = qtensor.insert %m2 into %rest2[%c2] : tensor<3x!qco.qubit>
    cbit.store %b0, %c[%seven] : !cbit.reg<9>
    cbit.store %b1, %c[%c2] : !cbit.reg<9>
    cbit.store %b2, %c[%five] : !cbit.reg<9>
    qtensor.dealloc %back2 : tensor<3x!qco.qubit>
    return %c : !cbit.reg<9>)mlir");
  auto source = parse(sourceText);
  auto compiled = parse(fixture("measured.mlir"));
  ASSERT_TRUE(source);
  ASSERT_TRUE(compiled);
  for (const auto relation : {
           VerificationRelation::Exact,
           VerificationRelation::TerminalProbabilities,
       }) {
    VerificationOptions options;
    options.method = VerificationMethod::Construction;
    options.relation = relation;
    auto result =
        qco::verify(entry(*source), entry(*compiled), {}, DDArgumentBindings(),
                    DDArgumentBindings(), options);
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->verdict, VerificationVerdict::Equivalent)
        << result->reason;
    ASSERT_EQ(result->outputLabels.size(), 9);
    EXPECT_EQ(result->outputLabels[2], "result0:c[2]");
    EXPECT_EQ(result->outputLabels[5], "result0:c[5]");
    EXPECT_EQ(result->outputLabels[7], "result0:c[7]");
    for (const auto& [original, mutation] :
         std::vector<std::pair<std::string, std::string>>{
             {"mqt.register_name = \"c\"", "mqt.register_name = \"other\""},
             {"arith.constant 7 : index", "arith.constant 8 : index"},
             {"cbit.store %b, %c[%c7]", "cbit.store %b, %c[%c5]"},
         }) {
      SCOPED_TRACE(original);
      auto mutated =
          parse(replace(fixture("measured.mlir"), original, mutation));
      ASSERT_TRUE(mutated);
      result = qco::verify(entry(*source), entry(*mutated), {},
                           DDArgumentBindings(), DDArgumentBindings(), options);
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(result->verdict, VerificationVerdict::NotEquivalent)
          << result->reason;
    }
  }
}

} // namespace
