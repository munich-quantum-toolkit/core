/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestUtils.hpp"

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

TEST(CompilerCLI, ValidatesMappingOptionArguments) {
  struct InvalidOptions {
    llvm::StringRef argument;
    bool hasDevice;
    llvm::StringRef diagnostic;
  };
  for (const auto& test : {
           InvalidOptions{
               .argument = "--mapping-trials=0",
               .hasDevice = false,
               .diagnostic = "Mapping controls require --qdmi-device",
           },
           InvalidOptions{
               .argument = "--mapping-iterations=0",
               .hasDevice = false,
               .diagnostic = "Mapping controls require --qdmi-device",
           },
           InvalidOptions{
               .argument = "--mapping-lookahead=0",
               .hasDevice = false,
               .diagnostic = "Mapping controls require --qdmi-device",
           },
           InvalidOptions{
               .argument = "--mapping-search-memory-limit=0",
               .hasDevice = false,
               .diagnostic = "Mapping controls require --qdmi-device",
           },
           InvalidOptions{
               .argument = "--mapping-trials=0",
               .hasDevice = true,
               .diagnostic = "--mapping-trials must be greater than zero",
           },
           InvalidOptions{
               .argument = "--mapping-iterations=0",
               .hasDevice = true,
               .diagnostic = "--qdmi-device or --target and --payload-spec "
                             "must be provided together",
           },
       }) {
    SCOPED_TRACE(test.argument.str());
    SCOPED_TRACE(test.hasDevice);
    llvm::SmallString<128> stderrPath;
    ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("mqt-cc-options", "err",
                                                    stderrPath));
    const llvm::FileRemover cleanup(stderrPath);
    llvm::SmallVector<llvm::StringRef> args{MQT_CORE_MQT_CC, test.argument};
    if (test.hasDevice) {
      args.push_back("--qdmi-device=mqt.ddsim.default");
    }
    EXPECT_EQ(llvm::sys::ExecuteAndWait(
                  MQT_CORE_MQT_CC, args, std::nullopt,
                  {std::nullopt, std::nullopt, stderrPath.str()}, 10),
              1);
    auto diagnostics = llvm::MemoryBuffer::getFile(stderrPath);
    ASSERT_TRUE(diagnostics);
    EXPECT_TRUE((*diagnostics)->getBuffer().contains(test.diagnostic));
  }
}

TEST(CompilerCLI, ValidatesExplicitTarget) {
  const llvm::StringRef target =
      "--target=#mqt.compilation_target<sites = [<id = 0>, <id = 1>], "
      "connectivity = all_to_all, couplings = [], native_operations = "
      "unrestricted, operations = []>";
  const llvm::StringRef payload =
      "--payload-spec=#mqt.payload_spec<format = <id = \"openqasm\", "
      "version = \"3.1.0\", profile = \"\", encoding = text>, capabilities = "
      "[], optional_capabilities_known = false>";
  struct InvalidOptions {
    llvm::SmallVector<llvm::StringRef> arguments;
    llvm::StringRef diagnostic;
  };
  for (const auto& test : {
           InvalidOptions{
               .arguments = {"--target=invalid", payload},
               .diagnostic = "--target must be a valid",
           },
           InvalidOptions{
               .arguments = {"--target=", payload},
               .diagnostic = "--target must be a valid",
           },
           InvalidOptions{
               .arguments = {"--target= \t", payload},
               .diagnostic = "--target must be a valid",
           },
           InvalidOptions{
               .arguments = {"--target=42 : i64", payload},
               .diagnostic = "--target must be a valid",
           },
           InvalidOptions{
               .arguments =
                   {
                       "--target=#mqt.compilation_target<sites = [], "
                       "connectivity = all_to_all, couplings = [], "
                       "native_operations = unrestricted, operations = []>",
                       payload,
                   },
               .diagnostic = "must contain at least one site",
           },
           InvalidOptions{
               .arguments = {target, payload, "--qdmi-device=mqt.sc.default"},
               .diagnostic = "mutually exclusive",
           },
           InvalidOptions{
               .arguments = {target},
               .diagnostic = "must be provided together",
           },
           InvalidOptions{
               .arguments = {payload},
               .diagnostic = "must be provided together",
           },
           InvalidOptions{
               .arguments = {target, payload, "--emit=qc"},
               .diagnostic = "Only --emit=qco-optimized",
           },
           InvalidOptions{
               .arguments = {target, payload, "--qdmi-list-devices"},
               .diagnostic = "cannot be combined with target compilation",
           },
           InvalidOptions{
               .arguments = {target, payload, "--pass-pipeline=canonicalize"},
               .diagnostic = "cannot be combined with --pass-pipeline",
           },
           InvalidOptions{
               .arguments = {target, payload, "--decompose-multi-controlled"},
               .diagnostic =
                   "cannot be combined with --decompose-multi-controlled",
           },
           InvalidOptions{
               .arguments = {target, payload, "--run-reproducer"},
               .diagnostic = "without --emit, target compilation",
           },
       }) {
    SCOPED_TRACE(test.diagnostic.str());
    llvm::SmallString<128> stderrPath;
    ASSERT_FALSE(
        llvm::sys::fs::createTemporaryFile("mqt-cc-target", "err", stderrPath));
    const llvm::FileRemover cleanup(stderrPath);
    llvm::SmallVector<llvm::StringRef> args{
        MQT_CORE_MQT_CC,
        MQT_CORE_MQT_CC_INPUT,
    };
    args.append(test.arguments);
    EXPECT_EQ(llvm::sys::ExecuteAndWait(
                  MQT_CORE_MQT_CC, args, std::nullopt,
                  {std::nullopt, std::nullopt, stderrPath.str()}, 10),
              1);
    auto diagnostics = llvm::MemoryBuffer::getFile(stderrPath);
    ASSERT_TRUE(diagnostics);
    EXPECT_TRUE((*diagnostics)->getBuffer().contains(test.diagnostic))
        << (*diagnostics)->getBuffer().str();
  }
}

TEST(CompilerCLI, CompilesForExplicitTarget) {
  llvm::SmallString<128> outputPath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("mqt-cc-target", "mlir", outputPath));
  const llvm::FileRemover cleanup(outputPath);
  const llvm::StringRef target =
      "--target=#mqt.compilation_target<sites = [<id = 0>, <id = 1>], "
      "connectivity = all_to_all, couplings = [], native_operations = "
      "unrestricted, operations = []>";
  const llvm::StringRef payload =
      "--payload-spec=#mqt.payload_spec<format = <id = \"openqasm\", "
      "version = \"3.1.0\", profile = \"\", encoding = text>, capabilities = "
      "[], optional_capabilities_known = false>";
  llvm::SmallVector<llvm::StringRef> args{
      MQT_CORE_MQT_CC,
      MQT_CORE_MQT_CC_INPUT,
      "--emit=qco-optimized",
      "--mapping-trials=1",
      target,
      payload,
  };
  ASSERT_EQ(llvm::sys::ExecuteAndWait(
                MQT_CORE_MQT_CC, args, std::nullopt,
                {std::nullopt, outputPath.str(), std::nullopt}, 10),
            0);
  auto output = llvm::MemoryBuffer::getFile(outputPath);
  ASSERT_TRUE(output);
  EXPECT_TRUE((*output)->getBuffer().contains("mqt.target_env"));
  EXPECT_TRUE((*output)->getBuffer().contains("mqt.layout"));
}

TEST(CompilerCLI, SeedOverridesCustomPassWithoutDevice) {
  std::string expected;
  for (bool useGlobalSeed : {false, true}) {
    llvm::SmallString<128> outputPath;
    ASSERT_FALSE(
        llvm::sys::fs::createTemporaryFile("mqt-cc-seed", "mlir", outputPath));
    const llvm::FileRemover cleanup(outputPath);
    llvm::SmallVector<llvm::StringRef> args{
        MQT_CORE_MQT_CC,
        MQT_CORE_MQT_CC_INPUT,
        "--emit=qco-optimized",
        "-o",
        outputPath,
        useGlobalSeed
            ? "--pass-pipeline=builtin.module(pauli-twirl-2q-gates{seed=99})"
            : "--pass-pipeline=builtin.module(pauli-twirl-2q-gates{seed=7})",
    };
    if (useGlobalSeed) {
      args.push_back("--seed=7");
    }
    ASSERT_EQ(
        llvm::sys::ExecuteAndWait(MQT_CORE_MQT_CC, args, std::nullopt, {}, 10),
        0);
    auto output = llvm::MemoryBuffer::getFile(outputPath);
    ASSERT_TRUE(output);
    if (useGlobalSeed) {
      EXPECT_EQ((*output)->getBuffer(), expected);
    } else {
      expected = (*output)->getBuffer().str();
    }
  }
}

TEST(CompilerCLI, PreservesLayoutAtImportCheckpoint) {
  llvm::SmallString<128> outputPath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("mqt-cc-layout", "mlir", outputPath));
  const llvm::FileRemover cleanup(outputPath);
  llvm::SmallVector<llvm::StringRef> args{
      MQT_CORE_MQT_CC,
      MQT_CORE_MQT_CC_LAYOUT_INPUT,
      "--emit=qc-import",
  };
  ASSERT_EQ(llvm::sys::ExecuteAndWait(
                MQT_CORE_MQT_CC, args, std::nullopt,
                {std::nullopt, outputPath.str(), std::nullopt}, 10),
            0);
  auto output = llvm::MemoryBuffer::getFile(outputPath);
  ASSERT_TRUE(output);
  EXPECT_TRUE((*output)->getBuffer().contains("mqt.layout"));
}

TEST(CompilerCLI, MappingTracePreservesOutputAndDescribesRouting) {
  const mqt::test::TemporaryDirectory directory;
  const auto input = directory
                         .write("triangle.qasm", R"(
OPENQASM 3.1;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
rx(0.3) q[0];
cz q[1], q[2];
cx q[0], q[2];
bit[3] result = measure q;
)")
                         .string();
  for (const auto completeGraph : {false, true}) {
    SCOPED_TRACE(completeGraph);
    const std::string target =
        "--target=#mqt.compilation_target<sites = [<id = 0>, <id = 1>, "
        "<id = 2>], connectivity = explicit, couplings = "
        "[<source = 0, target = 1>, <source = 1, target = 2>" +
        std::string(completeGraph ? ", <source = 0, target = 2>" : "") +
        "], native_operations = unrestricted, operations = []>";
    std::string reference;
    for (const auto trace : {false, true}) {
      const mqt::test::ScopedEnvironmentVariable environment(
          "MQT_MQSF_MAPPING_TRACE",
          trace ? std::optional<std::string>{"1"} : std::nullopt);
      // LLVM subprocess redirects do not truncate pre-existing files.
      const auto outputPath = directory.write("output.mlir", "").string();
      const auto stderrPath = directory.write("stderr.txt", "").string();
      constexpr llvm::StringRef payload =
          "--payload-spec=#mqt.payload_spec<format = <id = \"openqasm\", "
          "version = \"3.1.0\", profile = \"\", encoding = text>, "
          "capabilities = [], optional_capabilities_known = false>";
      const llvm::SmallVector<llvm::StringRef> args{
          MQT_CORE_MQT_CC,
          input,
          "--emit=qco-optimized",
          "--mapping-trials=2",
          "--mapping-iterations=1",
          "--seed=17",
          "--mlir-disable-threading",
          target,
          payload,
      };
      ASSERT_EQ(
          llvm::sys::ExecuteAndWait(MQT_CORE_MQT_CC, args, std::nullopt,
                                    {std::nullopt, outputPath, stderrPath}, 20),
          0);
      auto output = llvm::MemoryBuffer::getFile(outputPath);
      auto diagnostics = llvm::MemoryBuffer::getFile(stderrPath);
      ASSERT_TRUE(output);
      ASSERT_TRUE(diagnostics);
      if (!trace) {
        reference = (*output)->getBuffer().str();
        EXPECT_TRUE((*diagnostics)->getBuffer().empty())
            << (*diagnostics)->getBuffer().str();
        continue;
      }
      EXPECT_EQ((*output)->getBuffer(), reference);

      llvm::SmallVector<llvm::StringRef> lines;
      (*diagnostics)->getBuffer().split(lines, '\n', -1, false);
      std::set<std::string> phases;
      bool sawControlledX = false;
      bool sawRotation = false;
      for (auto line : lines) {
        ASSERT_TRUE(line.consume_front("MQSF_MAPPING_TRACE ")) << line.str();
        const auto event = nlohmann::json::parse(line.begin(), line.end());
        phases.insert(event.at("phase").get<std::string>());
        if (!event.contains("steps")) {
          continue;
        }
        auto layout = event.at("before").get<std::vector<size_t>>();
        size_t swaps = 0;
        for (const auto& step : event.at("steps")) {
          const auto sites = step.at("sites").get<std::vector<size_t>>();
          const auto logical = step.at("logical").get<std::vector<size_t>>();
          ASSERT_EQ(sites.size(), logical.size());
          for (size_t i = 0; i < sites.size(); ++i) {
            ASSERT_LT(logical[i], layout.size());
            EXPECT_EQ(layout[logical[i]], sites[i]);
          }
          if (sites.size() == 2 && !completeGraph) {
            EXPECT_EQ(std::max(sites[0], sites[1]) -
                          std::min(sites[0], sites[1]),
                      1U);
          }
          if (step.at("operation") == "qco.swap") {
            ASSERT_EQ(logical.size(), 2U);
            std::swap(layout[logical[0]], layout[logical[1]]);
            ++swaps;
          } else if (step.at("operation") == "qco.ctrl") {
            EXPECT_EQ(step.at("controls"), 1);
            EXPECT_TRUE(step.at("base_gate") == "qco.x" ||
                        step.at("base_gate") == "qco.z");
            sawControlledX |= step.at("base_gate") == "qco.x";
          } else if (step.at("operation") == "qco.rx") {
            EXPECT_EQ(step.at("base_gate"), "qco.rx");
            ASSERT_EQ(step.at("parameters").size(), 1U);
            EXPECT_DOUBLE_EQ(step.at("parameters")[0].get<double>(), 0.3);
            sawRotation = true;
          }
        }
        EXPECT_EQ(layout, event.at("after").get<std::vector<size_t>>());
        EXPECT_EQ(swaps, event.at("swaps"));
        // A triangle cannot be embedded in a line without exchanging wires.
        if (completeGraph) {
          EXPECT_EQ(swaps, 0U);
        } else {
          EXPECT_GE(swaps, 1U);
        }

        const auto& search = event.at("search");
        if (completeGraph) {
          EXPECT_TRUE(search.empty());
          continue;
        }
        EXPECT_FALSE(search.at("fallback").get<bool>());
        EXPECT_FALSE(search.at("truncated").get<bool>());
        const auto& nodes = search.at("nodes");
        ASSERT_FALSE(nodes.empty());
        EXPECT_EQ(nodes.size(), search.at("generated"));
        EXPECT_TRUE(nodes[0].at("parent").is_null());
        for (size_t i = 1; i < nodes.size(); ++i) {
          const auto parent = nodes[i].at("parent").get<size_t>();
          ASSERT_LT(parent, i);
          auto expected = nodes[parent].at("layout").get<std::vector<size_t>>();
          const auto edge = nodes[i].at("swap").get<std::vector<size_t>>();
          ASSERT_EQ(edge.size(), 2U);
          for (auto& site : expected) {
            site = site == edge[0] ? edge[1] : site == edge[1] ? edge[0] : site;
          }
          EXPECT_EQ(expected, nodes[i].at("layout").get<std::vector<size_t>>());
        }
        const auto goal = search.at("goal").get<size_t>();
        ASSERT_LT(goal, nodes.size());
        const auto final = nodes[goal].at("layout").get<std::vector<size_t>>();
        const auto front = search.at("front").get<std::vector<size_t>>();
        ASSERT_EQ(front.size(), 2U);
        ASSERT_LT(front[0], final.size());
        ASSERT_LT(front[1], final.size());
        EXPECT_EQ(std::max(final[front[0]], final[front[1]]) -
                      std::min(final[front[0]], final[front[1]]),
                  1U);
      }
      EXPECT_TRUE(sawControlledX);
      EXPECT_TRUE(sawRotation);
      EXPECT_TRUE(phases.contains("final-routing"));
      if (completeGraph) {
        EXPECT_TRUE(phases.contains("greedy-perfect"));
      } else {
        for (const auto* phase : {"forward", "backward", "score", "selected"}) {
          EXPECT_TRUE(phases.contains(phase));
        }
      }
    }
  }
}
