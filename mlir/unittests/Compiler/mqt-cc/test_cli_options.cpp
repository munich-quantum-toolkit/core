/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "gtest/gtest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

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
  const llvm::StringRef payload =
      "--payload-spec=#mqt.payload_spec<format = <id = \"openqasm\", "
      "version = \"3.1.0\", profile = \"\", encoding = text>, capabilities = "
      "[], "
      "optional_capabilities_known = false>";
  for (const auto argument : {
           llvm::StringRef("--target=invalid"),
           llvm::StringRef("--qdmi-device=mqt.sc.default"),
       }) {
    llvm::SmallString<128> stderrPath;
    ASSERT_FALSE(
        llvm::sys::fs::createTemporaryFile("mqt-cc-target", "err", stderrPath));
    const llvm::FileRemover cleanup(stderrPath);
    llvm::SmallVector<llvm::StringRef> args{
        MQT_CORE_MQT_CC,
        MQT_CORE_MQT_CC_INPUT,
        "--target=invalid",
        payload,
    };
    if (argument.starts_with("--qdmi-device")) {
      args.push_back(argument);
    }
    EXPECT_EQ(llvm::sys::ExecuteAndWait(
                  MQT_CORE_MQT_CC, args, std::nullopt,
                  {std::nullopt, std::nullopt, stderrPath.str()}, 10),
              1);
    auto diagnostics = llvm::MemoryBuffer::getFile(stderrPath);
    ASSERT_TRUE(diagnostics);
    EXPECT_TRUE((*diagnostics)
                    ->getBuffer()
                    .contains(argument.starts_with("--qdmi-device")
                                  ? "mutually exclusive"
                                  : "--target must be a valid"));
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
