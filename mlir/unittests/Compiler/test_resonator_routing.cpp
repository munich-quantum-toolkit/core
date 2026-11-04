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

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

static CompilerTarget starTarget(int scenario) {
  using Capability = CompilerTarget::OperationCapability;
  using Tuple = CompilerTarget::SiteTuple;
  std::vector unary{
      llvm::cantFail(Tuple::create({5})),
      llvm::cantFail(Tuple::create({17})),
  };
  std::vector pairs{
      llvm::cantFail(Tuple::create({5, 2})),
      llvm::cantFail(Tuple::create({17, 2})),
  };
  std::vector reversePairs{
      llvm::cantFail(Tuple::create({2, 5})),
      llvm::cantFail(Tuple::create({2, 17})),
  };
  if (scenario == 2) {
    pairs.pop_back();
  }
  if (scenario == 3) {
    pairs.back() = llvm::cantFail(Tuple::create({17, 3}));
    unary.push_back(llvm::cantFail(Tuple::create({19})));
  }
  std::vector operations{
      llvm::cantFail(Capability::create("r", 1, 2, unary)),
      llvm::cantFail(Capability::create("measure", 1, 0, unary)),
      llvm::cantFail(Capability::create(
          "cz", 2, 0,
          scenario == 1 ? reversePairs
          : scenario == 3
              ? std::vector{llvm::cantFail(Tuple::create({17, 2})),
                            llvm::cantFail(Tuple::create({19, 3})),}
              : std::vector{llvm::cantFail(Tuple::create({5, 2})),
                            llvm::cantFail(Tuple::create({17, 2})),})),
      llvm::cantFail(Capability::create("move", 2, 0, pairs)),
      llvm::cantFail(
          Capability::create("barrier", Capability::Arity::variadic(1), 0)),
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
      CompilerTarget::NativeOperations::fromOperations(operations)));
}

TEST(ResonatorRouting,
     CompilesThroughVirtualConnectivityAndKeepsReadoutOnQubits) {
  for (const int scenario : {0, 1, 2, 3, 4}) {
    SCOPED_TRACE(scenario);
    const auto target = starTarget(scenario);
    EXPECT_EQ(target.placeableSites(),
              (scenario == 3 ? std::vector<int64_t>{5, 17, 19}
                             : std::vector<int64_t>{5, 17}));
    TargetEnvironment environment(
        target, llvm::cantFail(PayloadSpecification::create({.id = "iqm-json"},
                                                            {}, true)));
    const std::string source =
        scenario == 3
            ? "OPENQASM 3; include \"stdgates.inc\"; qubit[3] q; bit[3] c; h "
              "q[0]; cz q[0],q[1]; cz q[1],q[2]; c=measure q;"
        : scenario == 4
            ? "OPENQASM 3; include \"stdgates.inc\"; bit[2] c; h $5; swap "
              "$5,$17; c[0]=measure $5; c[1]=measure $17;"
            : "OPENQASM 3; include \"stdgates.inc\"; qubit[2] q; bit[2] c; h "
              "q[0]; h q[1]; cz q[0],q[1]; c=measure q;";
    auto output =
        runDefaultPipeline(CompilerInput(OpenQASMProgram(source)), environment);
    ASSERT_TRUE(static_cast<bool>(output))
        << llvm::toString(output.takeError());
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
        EXPECT_TRUE((scenario == 1 ? locus.front() : locus.back()) == "CR" ||
                    (scenario == 1 ? locus.front() : locus.back()) == "CR2");
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
    EXPECT_EQ(moves, scenario == 3 ? 4U : scenario == 4 ? 6U : 2U);
  }
}

TEST(ResonatorRouting, PlacementRequiresGateAndMeasurement) {
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
