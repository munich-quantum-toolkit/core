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

TEST(CompilerCLI, RejectsInvalidMappingOptions) {
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
               .diagnostic = "--mapping-iterations must be greater than zero",
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

TEST(CompilerCLI, LayoutExportRequiresExplicitDiscard) {
  llvm::SmallString<128> outputPath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("mqt-cc-layout", "out", outputPath));
  const llvm::FileRemover cleanupOutput(outputPath);
  llvm::SmallString<128> stderrPath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("mqt-cc-layout", "err", stderrPath));
  const llvm::FileRemover cleanupStderr(stderrPath);

  for (llvm::StringRef format : {
           "--emit=openqasm3",
           "--emit=qir-base",
           "--emit=qir-adaptive",
           "--emit=jeff",
       }) {
    SCOPED_TRACE(format.str());
    for (bool discardLayout : {false, true}) {
      SCOPED_TRACE(discardLayout);
      llvm::SmallVector<llvm::StringRef> args{
          MQT_CORE_MQT_CC, MQT_CORE_MQT_CC_LAYOUT_INPUT, format, "-o",
          outputPath,
      };
      if (discardLayout) {
        args.push_back("--discard-layout");
      }
      const auto result = llvm::sys::ExecuteAndWait(
          MQT_CORE_MQT_CC, args, std::nullopt,
          {std::nullopt, std::nullopt, stderrPath.str()}, 10);
      auto diagnostics = llvm::MemoryBuffer::getFile(stderrPath);
      ASSERT_TRUE(diagnostics);
      EXPECT_EQ(result, discardLayout ? 0 : 1)
          << (*diagnostics)->getBuffer().str();
      if (!discardLayout) {
        EXPECT_TRUE((*diagnostics)
                        ->getBuffer()
                        .contains("cannot preserve qubit layout"));
      }
    }
  }
}

TEST(CompilerCLI, TracksLayoutAcrossImportAndTransformations) {
  llvm::SmallString<128> outputPath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("mqt-cc-layout", "mlir", outputPath));
  const llvm::FileRemover cleanup(outputPath);

  for (llvm::StringRef mode :
       {"--emit=qc-import", "--emit=qco-optimized", "--run-pipeline"}) {
    SCOPED_TRACE(mode.str());
    llvm::SmallVector<llvm::StringRef> args{
        MQT_CORE_MQT_CC,
        MQT_CORE_MQT_CC_LAYOUT_INPUT,
        mode,
    };
    if (mode == "--run-pipeline") {
      args.push_back("--pass-pipeline=builtin.module(canonicalize)");
    }
    ASSERT_EQ(llvm::sys::ExecuteAndWait(
                  MQT_CORE_MQT_CC, args, std::nullopt,
                  {std::nullopt, outputPath.str(), std::nullopt}, 10),
              0);
    auto output = llvm::MemoryBuffer::getFile(outputPath);
    ASSERT_TRUE(output);
    EXPECT_TRUE((*output)->getBuffer().contains(
        mode == "--emit=qc-import" ? "mqt.layout ="
                                   : "mqt.layout_invalidated"));
  }
}
