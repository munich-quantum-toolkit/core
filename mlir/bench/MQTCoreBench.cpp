/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// Generates configured benchmarks and evaluates their results.

#include "MQTCoreBench.h"

#include "bench/JSON.hpp"
#include "mlir/Compiler/Programs.h"
#include "mlir/bench/Generate.h"

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
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

[[nodiscard]] static bool
validateOutputTarget(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory) {
    return true;
  }
  if (error) {
    llvm::errs() << "failed to inspect '" << path.string()
                 << "': " << error.message() << '\n';
    return false;
  }
  if (status.type() != std::filesystem::file_type::not_found) {
    llvm::errs() << "refusing to overwrite existing file '" << path.string()
                 << "'\n";
    return false;
  }
  return true;
}

namespace {
struct OpenSibling {
  std::filesystem::path path;
  int descriptor;
};
} // namespace

[[nodiscard]] static std::filesystem::path
siblingPath(const std::filesystem::path& finalPath,
            const std::string_view purpose) {
  const auto random =
      (static_cast<uint64_t>(llvm::sys::Process::GetRandomNumber()) << 32U) |
      llvm::sys::Process::GetRandomNumber();
  const auto name = finalPath.filename().string() + "." + std::string(purpose) +
                    "-" + llvm::utohexstr(random);
  return finalPath.parent_path() / name;
}

[[nodiscard]] static std::optional<OpenSibling>
createSibling(const std::filesystem::path& finalPath,
              const std::string_view purpose) {
  for (size_t attempt = 0; attempt < 32; ++attempt) {
    auto path = siblingPath(finalPath, purpose);
    int descriptor = -1;
    const auto error = llvm::sys::fs::openFileForWrite(
        path.string(), descriptor, llvm::sys::fs::CD_CreateNew);
    if (!error) {
      return OpenSibling{.path = std::move(path), .descriptor = descriptor};
    }
    if (error != std::errc::file_exists) {
      llvm::errs() << "failed to create a " << purpose << " file next to '"
                   << finalPath.string() << "': " << error.message() << '\n';
      return std::nullopt;
    }
  }
  llvm::errs() << "failed to choose a unique " << purpose << " file next to '"
               << finalPath.string() << "'\n";
  return std::nullopt;
}

[[nodiscard]] static std::optional<std::filesystem::path>
stageFile(const std::filesystem::path& finalPath,
          const std::string_view contents) {
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
    if (const auto cleanupError =
            llvm::sys::fs::remove(temporary->path.string())) {
      llvm::errs() << "failed to remove temporary file '"
                   << temporary->path.string()
                   << "': " << cleanupError.message() << '\n';
    }
    llvm::errs() << "failed to write temporary output for '"
                 << finalPath.string() << "': " << error.message() << '\n';
    return std::nullopt;
  }
  return std::move(temporary->path);
}

static void removeIfPresent(const std::optional<std::filesystem::path>& path) {
  if (path) {
    if (const auto error = llvm::sys::fs::remove(path->string());
        error && error != std::errc::no_such_file_or_directory) {
      llvm::errs() << "failed to remove temporary file '" << path->string()
                   << "': " << error.message() << '\n';
    }
  }
}

[[nodiscard]] static const char*
programExtension(const std::string_view format) {
  if (format == "qc") {
    return ".qc.mlir";
  }
  if (format == "jeff") {
    return ".jeff";
  }
  llvm::errs() << "unsupported output format '" << format << "'\n";
  return nullptr;
}

[[nodiscard]] static int publish(mqt::bench::GeneratedBenchmark generated,
                                 const std::string_view format,
                                 const std::filesystem::path& directory) {
  const auto* const extension = programExtension(format);
  if (extension == nullptr) {
    return 1;
  }
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    llvm::errs() << "failed to create output directory '" << directory.string()
                 << "': " << error.message() << '\n';
    return 1;
  }
  const auto directoryExists = std::filesystem::is_directory(directory, error);
  if (error) {
    llvm::errs() << "failed to inspect output directory '" << directory.string()
                 << "': " << error.message() << '\n';
    return 1;
  }
  if (!directoryExists) {
    llvm::errs() << "output path is not a directory: '" << directory.string()
                 << "'\n";
    return 1;
  }

  const auto baseName = generated.benchmarkId + "-" + generated.caseId;
  const auto programPath = directory / (baseName + extension);
  const auto manifestPath =
      directory / (baseName + "." + std::string(format) + ".manifest.json");
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

  std::optional<std::filesystem::path> temporaryProgram;
  std::optional<std::filesystem::path> temporaryManifest;
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
          temporaryProgram->string(), programPath.string())) {
    llvm::errs() << "failed to publish '" << programPath.string()
                 << "': " << linkError.message() << '\n';
    return 1;
  }
  if (const auto linkError = llvm::sys::fs::create_hard_link(
          temporaryManifest->string(), manifestPath.string())) {
    llvm::errs() << "failed to publish '" << manifestPath.string()
                 << "': " << linkError.message() << "; program remains at '"
                 << programPath.string()
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
      {.K = "manifest_path", .V = manifestPath.string()},
      {.K = "program_path", .V = programPath.string()},
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
  return publish(std::move(*generated), outputFormat,
                 std::filesystem::path(outputDirectory.getValue()));
}

int runMQTCoreBench(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(benchmarkOptions);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Generate and evaluate structured quantum benchmarks\n");

  if (listCommand) {
    llvm::outs() << mqt::bench::listBenchmarksJSON() << '\n';
    return 0;
  }
  if (describeCommand) {
    llvm::outs() << mqt::bench::describeBenchmarkJSON(benchmarkId) << '\n';
    return 0;
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
    llvm::outs() << mqt::bench::evaluateJSON(*manifest, *counts,
                                             manifestInputPath, countsSource)
                 << '\n';
    return 0;
  }
  llvm::errs() << "a command is required; use --help for usage\n";
  return 1;
}
