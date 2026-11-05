/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/CompilationOptions.h"
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "ExactUnitaryTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace {

struct Scenario {
  const char* name;
  std::vector<int64_t> qubits;
  std::vector<std::vector<int64_t>> moves;
  std::vector<std::vector<int64_t>> czs;
  const char* source;
  size_t expectedMoves = 2;
  bool reversed = false;
};

constexpr const char* BELL_SOURCE =
    "OPENQASM 3; include \"stdgates.inc\"; qubit[2] q; bit[2] c; "
    "h q[0]; h q[1]; cz q[0],q[1]; c=measure q;";
constexpr const char* EARLY_MEASUREMENT_SOURCE =
    "OPENQASM 3; include \"stdgates.inc\"; qubit[3] q; bit[3] c; "
    "h q[0]; c[0]=measure q[0]; h q[1]; h q[2]; cz q[1],q[2]; "
    "c[1]=measure q[1]; c[2]=measure q[2];";

const std::vector<Scenario> SCENARIOS{
    {"Star", {5, 17}, {{5, 2}, {17, 2}}, {{5, 2}, {17, 2}}, BELL_SOURCE},
    {
        "Reversed",
        {5, 17},
        {{5, 2}, {17, 2}},
        {{2, 5}, {2, 17}},
        BELL_SOURCE,
        2,
        true,
    },
    {"SingleMove", {5, 17}, {{5, 2}}, {{5, 2}, {17, 2}}, BELL_SOURCE},
    {
        "TwoResonators",
        {5, 17, 19},
        {{5, 2}, {17, 3}},
        {{17, 2}, {19, 3}},
        "OPENQASM 3; include \"stdgates.inc\"; qubit[3] q; bit[3] c; "
        "h q[0]; cz q[0],q[1]; cz q[1],q[2]; c=measure q;",
        4,
    },
    {
        "Swap",
        {5, 17},
        {{5, 2}, {17, 2}},
        {{5, 2}, {17, 2}},
        "OPENQASM 3; include \"stdgates.inc\"; bit[2] c; h $5; swap $5,$17; "
        "c[0]=measure $5; c[1]=measure $17;",
        6,
    },
    {
        "EarlyMeasurement",
        {5, 17, 19},
        {{5, 2}, {17, 3}},
        {{17, 2}, {19, 3}},
        EARLY_MEASUREMENT_SOURCE,
    },
    {
        "Isolated",
        {5, 17, 19},
        {{5, 2}, {17, 2}},
        {{5, 2}, {17, 2}},
        BELL_SOURCE,
    },
};

TEST(NativeSynthesis, PreservesLogicalWireOrder) {
  auto qc = QCProgram::fromOpenQASMString(
      "OPENQASM 3.1; include \"stdgates.inc\"; qubit[4] q; "
      "x q[2]; h q[0]; swap q[0], q[2];");
  ASSERT_TRUE(static_cast<bool>(qc)) << llvm::toString(qc.takeError());
  auto program = std::move(*qc).intoQCO();
  ASSERT_TRUE(succeeded(program));
  auto expected = program->copy();
  const auto payload = llvm::cantFail(
      PayloadSpecification::forProgramFormat(ProgramFormat::OpenQASM3));
  const auto makeEnvironment = [&](size_t width) {
    const auto target = llvm::cantFail(CompilerTarget::create(
        width, CompilerTarget::Connectivity::allToAll(),
        CompilerTarget::NativeOperations::unrestricted()));
    return TargetEnvironment(target, payload);
  };
  const auto environment = makeEnvironment(4);
  CompilationOptions options;
  options.preserveLogicalOrder = true;
  ASSERT_TRUE(succeeded(program->synthesizeForTarget(environment, options)));
  EXPECT_FALSE(program->module()->hasAttr("mqt.layout"));
  ::mqt::test::expectFullUnitaryEqual(expected.module(), program->module(), 4);

  auto wider = expected.copy();
  EXPECT_TRUE(failed(wider.synthesizeForTarget(makeEnvironment(5), options)));
  EXPECT_TRUE(failed(program->synthesizeForTarget(environment, options)));
  EXPECT_TRUE(failed(expected.compileForTarget(environment, options)));
}

} // namespace

static CompilerTarget starTarget(const Scenario& scenario) {
  using Capability = CompilerTarget::OperationCapability;
  using Tuple = CompilerTarget::SiteTuple;
  std::vector<Tuple> unary;
  unary.reserve(scenario.qubits.size());
  for (const auto site : scenario.qubits) {
    unary.push_back(llvm::cantFail(Tuple::create({site})));
  }
  const auto tuples = [](const auto& pairs) {
    std::vector<Tuple> result;
    result.reserve(pairs.size());
    for (const auto& pair : pairs) {
      result.push_back(llvm::cantFail(Tuple::create(pair)));
    }
    return result;
  };
  return llvm::cantFail(CompilerTarget::create(
      "star",
      {
          llvm::cantFail(CompilerTarget::Site::create(2, "CR")),
          llvm::cantFail(CompilerTarget::Site::create(3, "CR2")),
          llvm::cantFail(CompilerTarget::Site::create(4, "UNCALIBRATED")),
          llvm::cantFail(CompilerTarget::Site::create(5, "QB1")),
          llvm::cantFail(CompilerTarget::Site::create(17, "QB2")),
          llvm::cantFail(CompilerTarget::Site::create(19, "QB3")),
      },
      CompilerTarget::Connectivity::fromCouplings(
          {{5, 2}, {17, 2}, {17, 3}, {19, 3}, {4, 2}}),
      CompilerTarget::NativeOperations::fromOperations({
          llvm::cantFail(Capability::create("r", 1, 2, unary)),
          llvm::cantFail(Capability::create("measure", 1, 0, unary)),
          llvm::cantFail(Capability::create("cz", 2, 0, tuples(scenario.czs))),
          llvm::cantFail(
              Capability::create("move", 2, 0, tuples(scenario.moves))),
          llvm::cantFail(
              Capability::create("barrier", Capability::Arity::variadic(1), 0)),
      })));
}

namespace {

class ResonatorRouting : public testing::TestWithParam<Scenario> {};

TEST_P(ResonatorRouting,
       CompilesThroughVirtualConnectivityAndKeepsReadoutOnQubits) {
  const auto& scenario = GetParam();
  const auto target = starTarget(scenario);
  EXPECT_EQ(target.placeableSites(), scenario.qubits);
  TargetEnvironment environment(
      target, llvm::cantFail(
                  PayloadSpecification::create({.id = "iqm-json"}, {}, true)));
  auto output = runDefaultPipeline(
      CompilerInput(OpenQASMProgram(scenario.source)), environment);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  const auto value = llvm::cantFail(
      llvm::json::parse(std::get<IQMJSONProgram>(*output).source()));
  const auto& instructions = *value.getAsObject()->getArray("instructions");
  size_t moves = 0;
  bool measured = false;
  for (size_t index = 0; index < instructions.size(); ++index) {
    const auto& instruction = *instructions[index].getAsObject();
    const auto name = instruction.getString("name");
    if (name == "move") {
      EXPECT_FALSE(measured);
      ++moves;
      EXPECT_TRUE(instruction.getArray("locus")->back() == "CR" ||
                  instruction.getArray("locus")->back() == "CR2");
    }
    if (name == "cz") {
      const auto& locus = *instruction.getArray("locus");
      EXPECT_TRUE((scenario.reversed ? locus.front() : locus.back()) == "CR" ||
                  (scenario.reversed ? locus.front() : locus.back()) == "CR2");
      ASSERT_GT(index, 0U);
      ASSERT_LT(index + 1, instructions.size());
      EXPECT_EQ(instructions[index - 1].getAsObject()->getString("name"),
                "move");
      EXPECT_EQ(instructions[index + 1].getAsObject()->getString("name"),
                "move");
      EXPECT_EQ(*instructions[index - 1].getAsObject()->getArray("locus"),
                *instructions[index + 1].getAsObject()->getArray("locus"));
    }
    if (name == "measure") {
      measured = true;
    }
    if (name == "measure" || name == "prx") {
      for (const auto& site : *instruction.getArray("locus")) {
        EXPECT_TRUE(site == "QB1" || site == "QB2" || site == "QB3");
      }
    }
  }
  EXPECT_EQ(moves, scenario.expectedMoves);
}

INSTANTIATE_TEST_SUITE_P(Placements, ResonatorRouting,
                         testing::ValuesIn(SCENARIOS),
                         [](const testing::TestParamInfo<Scenario>& info) {
                           return info.param.name;
                         });

TEST(ResonatorRouting, RawAdaptiveQIRSchedulesReadoutAfterMoves) {
  const TargetEnvironment environment(
      starTarget(SCENARIOS[5]),
      llvm::cantFail(
          PayloadSpecification::forProgramFormat(ProgramFormat::QIRAdaptive)));
  auto output = runDefaultPipeline(
      CompilerInput(OpenQASMProgram(EARLY_MEASUREMENT_SOURCE)),
      ProgramFormat::QIRAdaptive, environment);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  auto ir = std::move(std::get<QIRProgram>(*output)).intoLLVMIR();
  ASSERT_TRUE(succeeded(ir));
  const auto lastMove = ir->rfind("call void @__quantum__qis__move__body");
  const auto firstMeasure =
      ir->find("call void @__quantum__qis__measure__body");
  ASSERT_NE(lastMove, std::string::npos);
  ASSERT_NE(firstMeasure, std::string::npos) << *ir;
  EXPECT_LT(lastMove, firstMeasure);
}

TEST(ResonatorPlacement, PlacementRequiresGateAndMeasurement) {
  using Capability = CompilerTarget::OperationCapability;
  using Tuple = CompilerTarget::SiteTuple;
  for (const bool global : {false, true}) {
    const auto target = llvm::cantFail(CompilerTarget::create(
        3, CompilerTarget::Connectivity::allToAll(),
        CompilerTarget::NativeOperations::fromOperations({
            llvm::cantFail(Capability::create(
                "x", 1, 0,
                global ? std::vector<Tuple>{}
                       : std::vector{llvm::cantFail(Tuple::create({0})),
                                     llvm::cantFail(Tuple::create({1})),})),
            llvm::cantFail(
                Capability::create("measure", 1, 0,
                                   {llvm::cantFail(Tuple::create({0})),
                                    llvm::cantFail(Tuple::create({2})),})),
            llvm::cantFail(Capability::create("reset", 1, 0)),
            llvm::cantFail(Capability::create(
                "barrier", Capability::Arity::variadic(1), 0)),
        })));
    EXPECT_EQ(target.placeableSites(), global ? (std::vector<int64_t>{0, 2})
                                              : (std::vector<int64_t>{0}));
  }
}

TEST(ResonatorPlacement, KeepsLargestConnectedComponentAndDropsOversizedGates) {
  using Capability = CompilerTarget::OperationCapability;
  using Tuple = CompilerTarget::SiteTuple;
  const auto target = llvm::cantFail(CompilerTarget::create(
      4, CompilerTarget::Connectivity::fromCouplings({{0, 1}, {1, 2}, {2, 3}}),
      CompilerTarget::NativeOperations::fromOperations({
          llvm::cantFail(Capability::create("u", 1, 3)),
          llvm::cantFail(Capability::create("cx", 2, 0)),
          llvm::cantFail(Capability::create("ccx", 3, 0)),
          llvm::cantFail(
              Capability::create("measure", 1, 0,
                                 {
                                     llvm::cantFail(Tuple::create({0})),
                                     llvm::cantFail(Tuple::create({2})),
                                     llvm::cantFail(Tuple::create({3})),
                                 })),
      })));
  TargetEnvironment environment(
      target,
      llvm::cantFail(PayloadSpecification::create({.id = "qasm3"}, {}, true)));
  auto output = runDefaultPipeline(
      CompilerInput(OpenQASMProgram(
          "OPENQASM 3; include \"stdgates.inc\"; "
          "qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c=measure q;")),
      ProgramFormat::QC, environment);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  std::get<QCProgram>(*output).module().walk([](qc::StaticOp op) {
    EXPECT_TRUE(op.getIndex() == 2 || op.getIndex() == 3);
  });
}

TEST(ResonatorPlacement, RequiresCompilerTargetableUnarySignature) {
  using Capability = CompilerTarget::OperationCapability;
  using Tuple = CompilerTarget::SiteTuple;
  const auto target = llvm::cantFail(CompilerTarget::create(
      4, CompilerTarget::Connectivity::allToAll(),
      CompilerTarget::NativeOperations::fromOperations({
          llvm::cantFail(Capability::create(
              "x", 1, 0, {llvm::cantFail(Tuple::create({0}))})),
          llvm::cantFail(Capability::create(
              "h", 1, 0, {llvm::cantFail(Tuple::create({1}))})),
          llvm::cantFail(Capability::create(
              "vendor", 1, 0, {llvm::cantFail(Tuple::create({2}))})),
          llvm::cantFail(
              Capability::create("rx", 1, 0,
                                 {
                                     llvm::cantFail(Tuple::create({3})),
                                 })),
          llvm::cantFail(Capability::create("measure", 1, 0)),
      })));
  EXPECT_EQ(target.placeableSites(), (std::vector<int64_t>{0, 1}));
}

TEST(ResonatorPlacement, PreservesExistingMoveSymbol) {
  auto input = llvm::cantFail(runDefaultPipeline(
      CompilerInput(OpenQASMProgram(
          "OPENQASM 3; include \"stdgates.inc\"; "
          "qubit[2] q; bit[2] c; h q[0]; cz q[0],q[1]; c=measure q;")),
      ProgramFormat::QCO));
  auto program = std::get<QCOProgram>(std::move(input));
  auto module = program.module();
  OpBuilder builder(module.getContext());
  auto original = func::FuncOp::create(module.getLoc(), "move",
                                       builder.getFunctionType({}, {}));
  module.getBody()->push_back(original);
  builder.setInsertionPointToStart(original.addEntryBlock());
  func::ReturnOp::create(builder, module.getLoc());
  TargetEnvironment environment(
      starTarget(SCENARIOS.front()),
      llvm::cantFail(PayloadSpecification::create({.id = "qasm3"}, {}, true)));
  auto output = runDefaultPipeline(CompilerInput(std::move(program)),
                                   ProgramFormat::QC, environment);
  ASSERT_TRUE(static_cast<bool>(output)) << llvm::toString(output.takeError());
  auto result = std::get<QCProgram>(*output).module();
  EXPECT_TRUE(result.lookupSymbol<func::FuncOp>("move"));
  size_t calls = 0;
  result.walk([&](func::CallOp op) {
    EXPECT_NE(op.getCallee(), "move");
    ++calls;
  });
  EXPECT_EQ(calls, 2U);
}

} // namespace
