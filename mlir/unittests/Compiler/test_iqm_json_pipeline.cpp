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
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "support/JSON.hpp"

#include "gtest/gtest.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <string>
#include <utility>
#include <variant>

using namespace mlir;

static TargetEnvironment makeIQMEnvironment() {
  using Capability = CompilerTarget::OperationCapability;
  using Site = CompilerTarget::Site;
  const SmallVector capabilities{
      llvm::cantFail(Capability::create("r", 1, 2)),
      llvm::cantFail(Capability::create("cz", 2, 0)),
      llvm::cantFail(Capability::create("measure", 1, 0)),
  };
  auto target = llvm::cantFail(CompilerTarget::create(
      "sparse-iqm",
      {
          llvm::cantFail(Site::create(5, "readout-b")),
          llvm::cantFail(Site::create(17, "readout-a")),
          llvm::cantFail(Site::create(42, "spare")),
      },
      CompilerTarget::Connectivity::allToAll(),
      CompilerTarget::NativeOperations::fromOperations(capabilities)));
  auto payload = llvm::cantFail(
      PayloadSpecification::create({.id = "iqm-json"}, {}, true));
  return {target, std::move(payload)};
}

TEST(IQMJSONPipelineTest, CompilesInputFormatsForAnExplicitTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] result;
    r(1.25, -0.75) q[1];
    cz q[1], q[0];
    result[1] = measure q[1];
    result[0] = measure q[0];
  )qasm";
  auto qc = QCProgram::fromOpenQASMString(source.str());
  ASSERT_TRUE(succeeded(qc));
  EXPECT_TRUE(failed(
      runDefaultPipeline(CompilerInput(qc->copy()), ProgramFormat::IQMJSON)));
  auto qco = qc->copy().intoQCO();
  ASSERT_TRUE(succeeded(qco));
  auto jeff = qco->copy().intoJeff();
  ASSERT_TRUE(succeeded(jeff));
  SmallVector<CompilerInput> inputs;
  inputs.emplace_back(std::move(*qc));
  inputs.emplace_back(std::move(*qco));
  inputs.emplace_back(std::move(*jeff));
  inputs.emplace_back(OpenQASMProgram(source.str()));
  for (auto& input : inputs) {
    SCOPED_TRACE(input.index());
    auto output = runDefaultPipeline(std::move(input), makeIQMEnvironment());
    ASSERT_TRUE(succeeded(output));
    ASSERT_TRUE(std::holds_alternative<IQMJSONProgram>(*output));
    const auto& exported = std::get<IQMJSONProgram>(*output);
    const auto circuit = llvm::cantFail(::mqt::json::parse(exported.source()));
    const auto& instructions = *circuit.getAsObject()->getArray("instructions");
    EXPECT_EQ(instructions.front().getAsObject()->getString("name"), "prx");
    EXPECT_EQ(instructions.back().getAsObject()->getArray("locus")->size(), 1U);
    EXPECT_EQ(instructions.front().getAsObject()->getArray("locus")->front(),
              instructions.back().getAsObject()->getArray("locus")->front());
  }
}

TEST(IQMJSONPipelineTest, PreservesReversedStoresToDistinctOutputRegisters) {
  auto context = createCompilerContext();
  qc::QCProgramBuilder builder(context.get());
  builder.initialize();
  auto q0 = builder.allocQubit();
  auto q1 = builder.allocQubit();
  auto r0 = builder.allocClassicalBitRegister(1);
  auto r1 = builder.allocClassicalBitRegister(1);
  builder.x(q0);
  auto a = builder.measure(q0);
  auto b = builder.measure(q1);
  builder.storeClassicalBit(b, r1, 0);
  builder.storeClassicalBit(a, r0, 0);
  auto program = QCProgram::fromModule(context, builder.finalize({r0, r1}));
  ASSERT_TRUE(succeeded(program));
  auto output = runDefaultPipeline(CompilerInput(std::move(*program)),
                                   makeIQMEnvironment());
  ASSERT_TRUE(succeeded(output));
  const auto circuit = llvm::cantFail(
      ::mqt::json::parse(std::get<IQMJSONProgram>(*output).source()));
  const auto& instructions = *circuit.getAsObject()->getArray("instructions");
  ASSERT_GE(instructions.size(), 3U);
  const auto& firstOutput = instructions[instructions.size() - 2];
  EXPECT_EQ(firstOutput.getAsObject()->getString("name"), "measure");
  EXPECT_EQ(instructions.back().getAsObject()->getString("name"), "measure");
  EXPECT_EQ(instructions.front().getAsObject()->getString("name"), "prx");
  EXPECT_EQ(*instructions.front().getAsObject()->getArray("locus"),
            *firstOutput.getAsObject()->getArray("locus"));
  EXPECT_NE(*firstOutput.getAsObject()->getArray("locus"),
            *instructions.back().getAsObject()->getArray("locus"));
}

TEST(IQMJSONPipelineTest, UnrollsLoopsAndSynthesizesIQMBasis) {
  auto output = runDefaultPipeline(CompilerInput(OpenQASMProgram(R"qasm(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] result;
    for int i in [0:1] { h q[i]; }
    cx q[0], q[1];
    result = measure q;
  )qasm")),
                                   makeIQMEnvironment());
  ASSERT_TRUE(succeeded(output));
  const auto circuit = llvm::cantFail(
      ::mqt::json::parse(std::get<IQMJSONProgram>(*output).source()));
  const auto& instructions = *circuit.getAsObject()->getArray("instructions");
  size_t rotations = 0;
  size_t entanglers = 0;
  for (const auto& instruction : instructions) {
    const auto name = instruction.getAsObject()->getString("name")->str();
    EXPECT_TRUE(name == "prx" || name == "cz" || name == "measure") << name;
    rotations += static_cast<size_t>(name == "prx");
    entanglers += static_cast<size_t>(name == "cz");
  }
  EXPECT_GT(rotations, 0U);
  EXPECT_GT(entanglers, 0U);
  EXPECT_EQ(instructions.back().getAsObject()->getString("name"), "measure");
  EXPECT_EQ(instructions.back().getAsObject()->getArray("locus")->size(), 1U);
}

TEST(IQMJSONPipelineTest, RejectsReconstructedOutputsAndNonterminalGates) {
  for (const auto* source : {
           "qubit q; bit[2] result = 0; result[0] = measure q;",
           "qubit q; bit untouched = 0; bit result; result = measure q;",
           R"(qubit[2] q; bit result;
              result = measure q[0]; result = measure q[1];)",
           "qubit q; bit result; result = measure q; result = 0;",
           "qubit q; bit result; int value = 1; result = measure q;",
           R"(qubit[2] q; bit[2] result; h q[0]; h q[1];
              result[0] = measure q[0]; r(1, 0) q[0]; cz q[0], q[1];
              result[1] = measure q[1];)",
       }) {
    SCOPED_TRACE(source);
    auto input =
        QCProgram::fromOpenQASMString(std::string("OPENQASM 3.0; ") + source);
    ASSERT_TRUE(succeeded(input));
    ScopedDiagnosticHandler diagnostics(input->module().getContext(),
                                        [](Diagnostic&) { return success(); });
    EXPECT_TRUE(failed(runDefaultPipeline(CompilerInput(input->copy()),
                                          makeIQMEnvironment())));
  }
}

TEST(IQMJSONPipelineTest, DelaysIndependentReadoutUntilRoutedGatesFinish) {
  using Capability = CompilerTarget::OperationCapability;
  const auto target = llvm::cantFail(CompilerTarget::create(
      {
          llvm::cantFail(CompilerTarget::Site::create(10, "QB7")),
          llvm::cantFail(CompilerTarget::Site::create(30, "QB2")),
          llvm::cantFail(CompilerTarget::Site::create(90, "QB11")),
      },
      CompilerTarget::Connectivity::fromCouplings({{10, 30}, {30, 90}}),
      CompilerTarget::NativeOperations::fromOperations({
          llvm::cantFail(Capability::create("r", 1, 2)),
          llvm::cantFail(Capability::create("cz", 2, 0)),
          llvm::cantFail(Capability::create("measure", 1, 0)),
      })));
  const TargetEnvironment environment(
      target, makeIQMEnvironment().payloadSpecification());
  auto output = runDefaultPipeline(CompilerInput(OpenQASMProgram(R"qasm(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[2] a;
    bit a_0;
    x q[0];
    cx q[0], q[2];
    a[1] = measure q[0];
    a_0 = measure q[1];
    a[0] = measure q[2];
  )qasm")),
                                   environment);
  ASSERT_TRUE(succeeded(output));
  const auto circuit = llvm::cantFail(
      ::mqt::json::parse(std::get<IQMJSONProgram>(*output).source()));
  const auto& instructions = *circuit.getAsObject()->getArray("instructions");
  ASSERT_GT(instructions.size(), 3U);
  EXPECT_EQ(instructions[instructions.size() - 3]
                .getAsObject()
                ->getObject("args")
                ->getString("key"),
            "a_2_0_0");
  EXPECT_EQ(instructions[instructions.size() - 2]
                .getAsObject()
                ->getObject("args")
                ->getString("key"),
            "a_2_0_1");
  EXPECT_EQ(
      instructions.back().getAsObject()->getObject("args")->getString("key"),
      "a_0_1_1_0");
  for (size_t index = 0; index + 3 < instructions.size(); ++index) {
    EXPECT_NE(instructions[index].getAsObject()->getString("name"), "measure");
  }
}
