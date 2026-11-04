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
#include "mqt/Dialect/QIR/QIRDefinitions.h"
#include "qdmi/QDMI.hpp"

#include "NativeCommands.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

static llvm::Expected<qdmi::Job>
submit(const qdmi::Device& device, const std::string& path, int64_t shots,
       const std::optional<qdmi::CustomJobParameter>& seed) {
  const auto extension = llvm::sys::path::extension(path);
  if (extension == ".qasm") {
    auto program = mlir::QCProgram::fromOpenQASMFile(path);
    if (!program) {
      return program.takeError();
    }
    return mlir::submitProgram(device, std::move(*program), shots, std::nullopt,
                               seed);
  }
  if (extension == ".jeff") {
    auto program = mlir::JeffProgram::fromFile(path);
    if (!program) {
      return program.takeError();
    }
    return mlir::submitProgram(device, std::move(*program), shots, std::nullopt,
                               seed);
  }
  if (extension == ".mlir") {
    auto program = mlir::QCProgram::fromMLIRFile(path);
    if (!program) {
      return program.takeError();
    }
    return mlir::submitProgram(device, std::move(*program), shots, std::nullopt,
                               seed);
  }
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!buffer) {
    return llvm::errorCodeToError(buffer.getError());
  }
  const auto bytes = (*buffer)->getBuffer();
  if (extension == ".json") {
    return device.submitJob(bytes.str(), QDMI_PROGRAM_FORMAT_IQMJSON,
                            static_cast<size_t>(shots));
  }
  if (extension == ".qpy") {
    return device.submitJob(
        std::as_bytes(std::span(bytes.data(), bytes.size())),
        QDMI_PROGRAM_FORMAT_QPY, static_cast<size_t>(shots));
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module =
      llvm::parseIR((*buffer)->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("mqt-core", stream);
    return llvm::createStringError(message);
  }
  std::optional<bool> adaptive;
  for (const auto& function : *module) {
    if (function.isDeclaration() ||
        !function.hasFnAttribute(::qir::ENTRY_POINT_ATTR)) {
      continue;
    }
    if (!function.hasFnAttribute(::qir::QIR_PROFILES_ATTR)) {
      return llvm::createStringError("QIR entry point is missing qir_profiles");
    }
    const auto profile =
        function.getFnAttribute(::qir::QIR_PROFILES_ATTR).getValueAsString();
    if (adaptive || (profile != llvm::StringRef(::qir::BASE_PROFILE) &&
                     profile != llvm::StringRef(::qir::ADAPTIVE_PROFILE))) {
      return llvm::createStringError(
          "Expected one Base or Adaptive QIR entry point");
    }
    adaptive = profile == llvm::StringRef(::qir::ADAPTIVE_PROFILE);
  }
  if (!adaptive) {
    return llvm::createStringError("No QIR entry point found");
  }
  const bool binary = llvm::identify_magic(bytes) == llvm::file_magic::bitcode;
  const auto format = *adaptive
                          ? (binary ? QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE
                                    : QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING)
                          : (binary ? QDMI_PROGRAM_FORMAT_QIRBASEMODULE
                                    : QDMI_PROGRAM_FORMAT_QIRBASESTRING);
  if (!binary) {
    return device.submitJob(bytes.str(), format, static_cast<size_t>(shots),
                            seed);
  }
  return device.submitJob(std::as_bytes(std::span(bytes.data(), bytes.size())),
                          format, static_cast<size_t>(shots), seed);
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
  const std::optional<qdmi::CustomJobParameter> randomSeed =
      seed.getNumOccurrences() != 0
          ? std::optional<qdmi::CustomJobParameter>(seed.getValue())
          : std::nullopt;
  const auto job = exitOnError(submit(device, input, shots, randomSeed));
  exitOnError(job.wait());
  llvm::json::Object counts;
  for (const auto& [bits, count] : exitOnError(job.getCounts())) {
    counts[bits] = count;
  }
  llvm::outs() << llvm::json::Value(llvm::json::Object{
                      {"schema_version", 1},
                      {"counts", std::move(counts)},
                  })
               << '\n';
  return 0;
}
