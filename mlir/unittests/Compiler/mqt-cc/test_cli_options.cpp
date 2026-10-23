/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/JSON.hpp"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

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

TEST(CompilerCLI, EmitsTargetedIQMJSONAndPreservesOutputOnFailure) {
  llvm::SmallString<128> outputPath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("mqt-cc-iqm-target", "json",
                                                  outputPath));
  const llvm::FileRemover cleanup(outputPath);
  const llvm::StringRef target =
      "--target=#mqt.compilation_target<sites = "
      "[<id = 5, name = \"physical-b\">, <id = 17, name = \"physical-a\">], "
      "connectivity = all_to_all, couplings = [], native_operations = "
      "explicit, operations = [<name = \"r\", arity = "
      "#mqt.operation_arity<kind = fixed, value = 1>, num_parameters = 2, "
      "site_tuples = []>, "
      "<name = \"cz\", arity = #mqt.operation_arity<kind = fixed, value = 2>, "
      "num_parameters = 0, site_tuples = []>, "
      "<name = \"measure\", arity = #mqt.operation_arity<kind = fixed, value = "
      "1>, num_parameters = 0, site_tuples = []>]>";
  const llvm::StringRef payload =
      "--payload-spec=#mqt.payload_spec<format = <id = \"iqm-json\", "
      "version = \"\", profile = \"\", encoding = text>, capabilities = "
      "[], optional_capabilities_known = true>";
  llvm::SmallVector<llvm::StringRef> args{
      MQT_CORE_MQT_CC, MQT_CORE_MQT_CC_INPUT, target, payload, "-o", outputPath,
  };
  ASSERT_EQ(
      llvm::sys::ExecuteAndWait(MQT_CORE_MQT_CC, args, std::nullopt, {}, 10),
      0);
  auto output = llvm::MemoryBuffer::getFile(outputPath);
  ASSERT_TRUE(output);
  const auto circuit =
      llvm::cantFail(::mqt::json::parse((*output)->getBuffer().str()));
  const auto& instructions = *circuit.getAsObject()->getArray("instructions");
  EXPECT_EQ(instructions.back().getAsObject()->getString("name"), "measure");
  EXPECT_EQ(instructions.back().getAsObject()->getArray("locus")->size(), 1U);
  for (const auto& site :
       *instructions.back().getAsObject()->getArray("locus")) {
    EXPECT_TRUE(site == "physical-a" || site == "physical-b")
        << ::mqt::json::serialize(site);
  }
  args[2] = "--target=#mqt.compilation_target<sites = [<id = 5>, <id = 17>], "
            "connectivity = all_to_all, couplings = [], native_operations = "
            "unrestricted, operations = []>";
  EXPECT_EQ(
      llvm::sys::ExecuteAndWait(MQT_CORE_MQT_CC, args, std::nullopt, {}, 10),
      1);
  auto preserved = llvm::MemoryBuffer::getFile(outputPath);
  ASSERT_TRUE(preserved);
  EXPECT_EQ((*preserved)->getBuffer(), (*output)->getBuffer());
}
