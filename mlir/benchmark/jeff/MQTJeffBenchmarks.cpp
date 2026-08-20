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
 * @brief Generates `jeff` files for the structured benchmark programs.
 *
 * @details Each program is built with the `QCProgramBuilder` for a given size
 * and then lowered through QCO to `jeff`.
 */

#include "mlir/Benchmark/Compile.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Compiler/Programs.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

static llvm::cl::opt<uint64_t>
    numQubits("n", llvm::cl::desc("Size parameter of the generated programs"),
              llvm::cl::value_desc("n"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    outputDirectory("o", llvm::cl::desc("Directory for the generated files"),
                    llvm::cl::value_desc("directory"), llvm::cl::init("."));

static llvm::cl::opt<std::string> programFilter(
    "program",
    llvm::cl::desc("Generate only the named program instead of every one"),
    llvm::cl::value_desc("name"), llvm::cl::init(""));

/// Builds one benchmark and writes it as a `jeff` file.
static bool generate(const mqt::benchmark::Benchmark& benchmark,
                     const uint64_t n, const std::filesystem::path& directory) {
  if (n < benchmark.minimumSize) {
    llvm::errs() << benchmark.name << ": needs n of at least "
                 << benchmark.minimumSize << ", skipping\n";
    return false;
  }

  auto qc = mqt::benchmark::buildQCProgram(benchmark, n);
  if (!qc) {
    llvm::errs() << benchmark.name << ": failed to build the program\n";
    return false;
  }
  auto compiled =
      mlir::runDefaultPipeline(std::move(*qc), mlir::ProgramFormat::Jeff);
  if (!compiled) {
    llvm::errs() << benchmark.name << ": failed to build the jeff program\n";
    return false;
  }
  const auto& program = std::get<mlir::JeffProgram>(*compiled);

  const auto path = directory / (benchmark.name.str() + ".jeff");
  if (!program.write(path)) {
    llvm::errs() << benchmark.name << ": failed to write " << path.string()
                 << "\n";
    return false;
  }

  llvm::outs() << benchmark.name << " -> " << path.string() << "\n";
  return true;
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Generate jeff files for the structured benchmarks\n");

  const std::filesystem::path directory(outputDirectory.getValue());
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    llvm::errs() << "failed to create " << directory.string() << ": "
                 << error.message() << "\n";
    return 1;
  }

  const auto filter = llvm::StringRef(programFilter.getValue());
  auto failures = 0;
  auto generated = 0;
  for (const auto& benchmark : mqt::benchmark::benchmarks()) {
    if (!filter.empty() && benchmark.name != filter) {
      continue;
    }
    if (generate(benchmark, numQubits.getValue(), directory)) {
      ++generated;
    } else {
      ++failures;
    }
  }

  if (!filter.empty() && generated == 0 && failures == 0) {
    llvm::errs() << "no program named '" << filter << "'\n";
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
