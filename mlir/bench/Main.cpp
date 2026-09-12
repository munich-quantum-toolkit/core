/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "MQTCoreBench.h"

#include <llvm/Support/CommandLine.h>

#include <string>

static llvm::cl::OptionCategory benchmarkOptions("Benchmark options");
static llvm::cl::SubCommand listCommand("list",
                                        "List the available benchmarks");
static llvm::cl::SubCommand
    describeCommand("describe",
                    "Describe one benchmark instance specification schema");
static llvm::cl::SubCommand
    generateCommand("generate", "Generate one configured benchmark");
static llvm::cl::SubCommand
    evaluateCommand("evaluate", "Evaluate counts against a manifest");

static llvm::cl::opt<std::string> benchmarkId(llvm::cl::Positional,
                                              llvm::cl::desc("<id>"),
                                              llvm::cl::Required,
                                              llvm::cl::cat(benchmarkOptions),
                                              llvm::cl::sub(describeCommand));

static llvm::cl::opt<std::string> instanceSpecificationPath(
    "instance-specification",
    llvm::cl::desc(
        "Instance specification JSON file, or '-' for standard input"),
    llvm::cl::value_desc("file|-"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(generateCommand));

static llvm::cl::opt<std::string> outputFormat(
    "format", llvm::cl::desc("Generated program format: qc or jeff"),
    llvm::cl::value_desc("qc|jeff"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(generateCommand));

static llvm::cl::opt<std::string> outputDirectory(
    "output", llvm::cl::desc("Directory for the program and manifest"),
    llvm::cl::value_desc("directory"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(generateCommand));

static llvm::cl::opt<std::string> manifestInputPath(
    "manifest", llvm::cl::desc("Benchmark manifest JSON file"),
    llvm::cl::value_desc("file"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(evaluateCommand));

static llvm::cl::opt<std::string> countsInputPath(
    "counts", llvm::cl::desc("Counts JSON file, or '-' for standard input"),
    llvm::cl::value_desc("file|-"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(evaluateCommand));

int main(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(benchmarkOptions);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Generate and evaluate structured quantum benchmarks\n");

  auto command = BenchmarkOptions::Command::None;
  if (listCommand) {
    command = BenchmarkOptions::Command::List;
  } else if (describeCommand) {
    command = BenchmarkOptions::Command::Describe;
  } else if (generateCommand) {
    command = BenchmarkOptions::Command::Generate;
  } else if (evaluateCommand) {
    command = BenchmarkOptions::Command::Evaluate;
  }
  return runMQTCoreBench({
      .command = command,
      .benchmarkId = benchmarkId,
      .instanceSpecificationPath = instanceSpecificationPath,
      .outputFormat = outputFormat,
      .outputDirectory = outputDirectory,
      .manifestInputPath = manifestInputPath,
      .countsInputPath = countsInputPath,
  });
}
