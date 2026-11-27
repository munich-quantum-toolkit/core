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
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/ResultLayout.h"
#include "qdmi/QDMI.hpp"

#include "NativeCommands.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

static llvm::Expected<qdmi::Job>
submit(const qdmi::Device& device, const std::string& path,
       const qdmi::JobParameters& parameters,
       std::optional<mlir::ResultLayout>& layout,
       std::optional<size_t>& width) {
  const auto submitSource = [&](auto program) -> llvm::Expected<qdmi::Job> {
    if (!program) {
      return program.takeError();
    }
    auto compiled = mlir::compileProgram(std::move(*program), device);
    if (!compiled) {
      return compiled.takeError();
    }
    width = compiled->resultWidth();
    if (width.value_or(0) == 0) {
      return llvm::createStringError(
          "Counts require a nonempty classical bit output. "
          "Return measured bits to count their outcomes.");
    }
    layout = compiled->resultLayout();
    // The program was just compiled for this device snapshot.
    return mlir::submitProgram(device, *compiled, parameters,
                               &compiled->environment());
  };
  if (const auto format = mlir::inputFormatForPath(path)) {
    return submitSource(mlir::loadCompilerInput(path, format));
  }
  const auto extension = llvm::sys::path::extension(path);
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!buffer) {
    return llvm::errorCodeToError(buffer.getError());
  }
  const auto bytes = (*buffer)->getBuffer();
  const auto binary = std::as_bytes(std::span(bytes.data(), bytes.size()));
  if (extension == ".json") {
    return device.submitJob(bytes.str(), QDMI_PROGRAM_FORMAT_IQMJSON,
                            {.shots = parameters.shots});
  }
  if (extension == ".qpy") {
    return device.submitJob(binary, QDMI_PROGRAM_FORMAT_QPY,
                            {.shots = parameters.shots});
  }
  auto format = mlir::detectQIRFormat(bytes);
  if (!format) {
    return format.takeError();
  }
  if (qdmi::isBinaryProgramFormat(**format)) {
    return device.submitJob(binary, **format, parameters);
  }
  return device.submitJob(bytes.str(), **format, parameters);
}

int mqtCoreRunProgram(int argc, char** argv) {
  const bool simulate = llvm::StringRef(argv[0]) == "simulate";
  llvm::cl::OptionCategory category("Execution options");
  llvm::cl::opt<std::string> deviceId(
      "qdmi-device", llvm::cl::desc("QDMI device ID (required for run)"),
      llvm::cl::init(simulate ? "mqt.ddsim.default" : ""),
      llvm::cl::cat(category));
  llvm::cl::opt<std::string> input(
      llvm::cl::Positional, llvm::cl::Required,
      llvm::cl::desc(
          "<file.qasm|file.ll|file.bc|file.json|file.jeff|file.qpy|file.mlir>"),
      llvm::cl::cat(category));
  llvm::cl::opt<int64_t> shots("shots", llvm::cl::init(1024),
                               llvm::cl::desc("Number of shots"),
                               llvm::cl::cat(category));
  llvm::cl::opt<int> seed("seed", llvm::cl::desc("Random seed (DDSIM only)"),
                          llvm::cl::cat(category));
  llvm::cl::HideUnrelatedOptions(category);
  if (!llvm::cl::ParseCommandLineOptions(
          argc, argv,
          "Run a quantum program and print benchmark-compatible counts JSON\n",
          &llvm::errs())) {
    return 1;
  }
  llvm::ExitOnError exitOnError("mqt-core: ");
  if (deviceId.empty() || (simulate && deviceId != "mqt.ddsim.default") ||
      (seed.getNumOccurrences() != 0 && deviceId != "mqt.ddsim.default")) {
    llvm::errs()
        << "run requires --qdmi-device; simulate and --seed require DDSIM\n";
    return 1;
  }
  if (shots <= 0) {
    llvm::errs() << "mqt-core: --shots must be positive\n";
    return 1;
  }
  const auto device =
      exitOnError(qdmi::builtin_driver::openDevice(deviceId.getValue()));
  qdmi::JobParameters parameters{.shots = static_cast<size_t>(shots)};
  if (seed.getNumOccurrences() != 0) {
    parameters.custom[0] = seed.getValue();
  }
  std::optional<mlir::ResultLayout> layout;
  std::optional<size_t> width;
  const auto job =
      exitOnError(submit(device, input, parameters, layout, width));
  exitOnError(job.wait());
  llvm::json::Object counts;
  const auto decoded = exitOnError(mlir::decodeCounts(
      exitOnError(job.getCounts()), width, static_cast<size_t>(shots), layout));
  for (const auto& [bits, count] : decoded) {
    counts[bits] = static_cast<int64_t>(count);
  }
  llvm::outs() << llvm::json::Value(llvm::json::Object{
                      {"schema_version", 1},
                      {"counts", std::move(counts)},
                  })
               << '\n';
  return 0;
}
