/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// Generates configured benchmarks and evaluates their results.

#include "bench/JSON.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/bench/Generate.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

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

[[nodiscard]] static std::optional<std::string>
readText(const std::string& path) {
  auto buffer = path == "-" ? llvm::MemoryBuffer::getSTDIN()
                            : llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "failed to read '" << path
                 << "': " << buffer.getError().message() << '\n';
    return std::nullopt;
  }
  return (*buffer)->getBuffer().str();
}

[[nodiscard]] static bool validateOutputTarget(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  const auto error = llvm::sys::fs::status(path, status, false);
  if (error == std::errc::no_such_file_or_directory) {
    return true;
  }
  if (error) {
    llvm::errs() << "failed to inspect '" << path << "': " << error.message()
                 << '\n';
    return false;
  }
  if (llvm::sys::fs::exists(status)) {
    llvm::errs() << "refusing to overwrite existing file '" << path << "'\n";
    return false;
  }
  return true;
}

namespace {
struct OpenSibling {
  std::string path;
  int descriptor;
};
} // namespace

[[nodiscard]] static std::string siblingPath(llvm::StringRef finalPath,
                                             const std::string_view purpose) {
  const auto random =
      (static_cast<uint64_t>(llvm::sys::Process::GetRandomNumber()) << 32U) |
      llvm::sys::Process::GetRandomNumber();
  return finalPath.str() + "." + std::string(purpose) + "-" +
         llvm::utohexstr(random);
}

[[nodiscard]] static std::optional<OpenSibling>
createSibling(llvm::StringRef finalPath, const std::string_view purpose) {
  for (size_t attempt = 0; attempt < 32; ++attempt) {
    auto path = siblingPath(finalPath, purpose);
    int descriptor = -1;
    const auto error = llvm::sys::fs::openFileForWrite(
        path, descriptor, llvm::sys::fs::CD_CreateNew);
    if (!error) {
      return OpenSibling{.path = std::move(path), .descriptor = descriptor};
    }
    if (error != std::errc::file_exists) {
      llvm::errs() << "failed to create a " << purpose << " file next to '"
                   << finalPath << "': " << error.message() << '\n';
      return std::nullopt;
    }
  }
  llvm::errs() << "failed to choose a unique " << purpose << " file next to '"
               << finalPath << "'\n";
  return std::nullopt;
}

[[nodiscard]] static std::optional<std::string>
stageFile(llvm::StringRef finalPath, const std::string_view contents) {
  auto temporary = createSibling(finalPath, "tmp");
  if (!temporary) {
    return std::nullopt;
  }
  llvm::raw_fd_ostream stream(temporary->descriptor, true);
  stream.write(contents.data(), contents.size());
  stream.close();
  const auto error = stream.error();
  stream.clear_error();
  if (error) {
    if (const auto cleanupError = llvm::sys::fs::remove(temporary->path)) {
      llvm::errs() << "failed to remove temporary file '" << temporary->path
                   << "': " << cleanupError.message() << '\n';
    }
    llvm::errs() << "failed to write temporary output for '" << finalPath
                 << "': " << error.message() << '\n';
    return std::nullopt;
  }
  return std::move(temporary->path);
}

static void removeIfPresent(const std::optional<std::string>& path) {
  if (path) {
    if (const auto error = llvm::sys::fs::remove(*path);
        error && error != std::errc::no_such_file_or_directory) {
      llvm::errs() << "failed to remove temporary file '" << *path
                   << "': " << error.message() << '\n';
    }
  }
}

[[nodiscard]] static int publish(mqt::bench::GeneratedBenchmark generated,
                                 const std::string_view format,
                                 llvm::StringRef directory) {
  if (format != "qc" && format != "jeff") {
    llvm::errs() << "unsupported output format '" << format << "'\n";
    return 1;
  }
  const auto* const extension = format == "qc" ? ".qc.mlir" : ".jeff";
  if (const auto error = llvm::sys::fs::create_directories(directory)) {
    llvm::errs() << "failed to create output directory '" << directory
                 << "': " << error.message() << '\n';
    return 1;
  }
  bool directoryExists = false;
  if (const auto error =
          llvm::sys::fs::is_directory(directory, directoryExists)) {
    llvm::errs() << "failed to inspect output directory '" << directory
                 << "': " << error.message() << '\n';
    return 1;
  }
  if (!directoryExists) {
    llvm::errs() << "output path is not a directory: '" << directory << "'\n";
    return 1;
  }

  const auto baseName = generated.benchmarkId + "-" + generated.caseId;
  llvm::SmallString<128> programPath(directory);
  llvm::sys::path::append(programPath, baseName + extension);
  llvm::SmallString<128> manifestPath(directory);
  llvm::sys::path::append(manifestPath, baseName + "." + std::string(format) +
                                            ".manifest.json");
  if (!validateOutputTarget(programPath) ||
      !validateOutputTarget(manifestPath)) {
    return 1;
  }

  std::string serializedProgram;
  if (format == "qc") {
    serializedProgram = generated.program.str();
    if (serializedProgram.empty() || serializedProgram.back() != '\n') {
      serializedProgram.push_back('\n');
    }
  } else {
    auto compiled = mlir::runDefaultPipeline(std::move(generated.program),
                                             mlir::ProgramFormat::Jeff);
    if (!compiled) {
      llvm::errs() << generated.benchmarkId
                   << ": failed to build the jeff program\n";
      return 1;
    }
    const auto bytes = std::get<mlir::JeffProgram>(*compiled).toBytes();
    serializedProgram.assign(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size());
  }
  auto manifest = std::move(generated.manifestJSON);
  manifest.push_back('\n');

  std::optional<std::string> temporaryProgram;
  std::optional<std::string> temporaryManifest;
  const auto removeTemporaryFiles = llvm::make_scope_exit([&] {
    removeIfPresent(temporaryProgram);
    removeIfPresent(temporaryManifest);
  });

  temporaryProgram = stageFile(programPath, serializedProgram);
  if (!temporaryProgram) {
    return 1;
  }
  temporaryManifest = stageFile(manifestPath, manifest);
  if (!temporaryManifest) {
    return 1;
  }

  if (const auto linkError = llvm::sys::fs::create_hard_link(
          *temporaryProgram, programPath.str())) {
    llvm::errs() << "failed to publish '" << programPath.str()
                 << "': " << linkError.message() << '\n';
    return 1;
  }
  if (const auto linkError = llvm::sys::fs::create_hard_link(
          *temporaryManifest, manifestPath.str())) {
    llvm::errs() << "failed to publish '" << manifestPath.str()
                 << "': " << linkError.message() << "; program remains at '"
                 << programPath.str()
                 << "'; this invocation did not publish a manifest\n";
    return 1;
  }
  removeIfPresent(temporaryProgram);
  removeIfPresent(temporaryManifest);
  temporaryProgram.reset();
  temporaryManifest.reset();

  llvm::json::Object response{
      {.K = "benchmark", .V = generated.benchmarkId},
      {.K = "case_id", .V = generated.caseId},
      {.K = "format", .V = std::string(format)},
      {.K = "manifest_path", .V = manifestPath.str()},
      {.K = "program_path", .V = programPath.str()},
      {.K = "schema_version", .V = 1},
  };
  llvm::outs() << llvm::json::Value(std::move(response)) << '\n';
  return 0;
}

[[nodiscard]] static int
generateFromInstanceSpecification(const std::string& instanceSpecification,
                                  const std::string& source) {
  auto generated = mqt::bench::generate(instanceSpecification, source);
  if (!generated) {
    return 1;
  }
  return publish(std::move(*generated), outputFormat, outputDirectory);
}

[[nodiscard]] static int
printJSON(std::variant<std::string, mqt::bench::JSONError> result) {
  if (const auto* error = std::get_if<mqt::bench::JSONError>(&result)) {
    llvm::errs() << error->message << '\n';
    return 1;
  }
  llvm::outs() << std::get<std::string>(result) << '\n';
  return 0;
}

int main(int argc, char** argv) {
  const llvm::InitLLVM init(argc, argv);
  llvm::cl::HideUnrelatedOptions(benchmarkOptions);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Generate and evaluate structured quantum benchmarks\n");
  if (listCommand) {
    llvm::outs() << mqt::bench::listBenchmarksJSON() << '\n';
    return 0;
  }
  if (describeCommand) {
    return printJSON(mqt::bench::tryDescribeBenchmarkJSON(benchmarkId));
  }
  if (generateCommand) {
    const auto instanceSpecification = readText(instanceSpecificationPath);
    if (!instanceSpecification) {
      return 1;
    }
    const auto source = instanceSpecificationPath == "-"
                            ? "<stdin>"
                            : instanceSpecificationPath.getValue();
    return generateFromInstanceSpecification(*instanceSpecification, source);
  }
  if (evaluateCommand) {
    if (manifestInputPath == "-") {
      llvm::errs() << "--manifest requires a file path\n";
      return 1;
    }
    const auto manifest = readText(manifestInputPath);
    if (!manifest) {
      return 1;
    }
    const auto counts = readText(countsInputPath);
    if (!counts) {
      return 1;
    }
    const auto countsSource =
        countsInputPath == "-" ? "<stdin>" : countsInputPath.getValue();
    return printJSON(mqt::bench::tryEvaluateJSON(
        *manifest, *counts, manifestInputPath, countsSource));
  }
  llvm::errs() << "a command is required; use --help for usage\n";
  return 1;
}
