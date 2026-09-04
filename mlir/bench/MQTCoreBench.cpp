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
#include <stdexcept>
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

[[nodiscard]] static std::string readText(const std::string& path) {
  auto buffer = path == "-" ? llvm::MemoryBuffer::getSTDIN()
                            : llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    throw std::runtime_error("failed to read '" + path +
                             "': " + buffer.getError().message());
  }
  return (*buffer)->getBuffer().str();
}

[[nodiscard]] static bool pathExists(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory) {
    return false;
  }
  if (error) {
    throw std::runtime_error("failed to inspect '" + path.string() +
                             "': " + error.message());
  }
  return status.type() != std::filesystem::file_type::not_found;
}

static void validateOutputTarget(const std::filesystem::path& path) {
  if (pathExists(path)) {
    throw std::runtime_error("refusing to overwrite existing file '" +
                             path.string() + "'");
  }
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

[[nodiscard]] static OpenSibling
createSibling(const std::filesystem::path& finalPath,
              const std::string_view purpose) {
  for (size_t attempt = 0; attempt < 32; ++attempt) {
    auto path = siblingPath(finalPath, purpose);
    int descriptor = -1;
    const auto error = llvm::sys::fs::openFileForWrite(
        path.string(), descriptor, llvm::sys::fs::CD_CreateNew);
    if (!error) {
      return {.path = std::move(path), .descriptor = descriptor};
    }
    if (error != std::errc::file_exists) {
      throw std::runtime_error("failed to create a " + std::string(purpose) +
                               " file next to '" + finalPath.string() +
                               "': " + error.message());
    }
  }
  throw std::runtime_error("failed to choose a unique " + std::string(purpose) +
                           " file next to '" + finalPath.string() + "'");
}

[[nodiscard]] static std::filesystem::path
stageFile(const std::filesystem::path& finalPath,
          const std::string_view contents) {
  auto temporary = createSibling(finalPath, "tmp");
  llvm::raw_fd_ostream stream(temporary.descriptor, true);
  stream.write(contents.data(), contents.size());
  stream.close();
  const auto error = stream.error();
  stream.clear_error();
  if (error) {
    if (const auto cleanupError =
            llvm::sys::fs::remove(temporary.path.string())) {
      llvm::errs() << "failed to remove temporary file '"
                   << temporary.path.string() << "': " << cleanupError.message()
                   << '\n';
    }
    throw std::runtime_error("failed to write temporary output for '" +
                             finalPath.string() + "': " + error.message());
  }
  return temporary.path;
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
  throw std::invalid_argument("unsupported output format '" +
                              std::string(format) + "'");
}

[[nodiscard]] static int publish(mqt::bench::GeneratedBenchmark generated,
                                 const std::string_view format,
                                 const std::filesystem::path& directory) {
  const auto extension = programExtension(format);
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    throw std::runtime_error("failed to create output directory '" +
                             directory.string() + "': " + error.message());
  }
  const auto directoryExists = std::filesystem::is_directory(directory, error);
  if (error) {
    throw std::runtime_error("failed to inspect output directory '" +
                             directory.string() + "': " + error.message());
  }
  if (!directoryExists) {
    throw std::runtime_error("output path is not a directory: '" +
                             directory.string() + "'");
  }

  const auto baseName = generated.benchmarkId + "-" + generated.caseId;
  const auto programPath = directory / (baseName + extension);
  const auto manifestPath =
      directory / (baseName + "." + std::string(format) + ".manifest.json");
  validateOutputTarget(programPath);
  validateOutputTarget(manifestPath);

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
  temporaryManifest = stageFile(manifestPath, manifest);

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
      {"benchmark", generated.benchmarkId},
      {"case_id", generated.caseId},
      {"format", std::string(format)},
      {"manifest_path", manifestPath.string()},
      {"program_path", programPath.string()},
      {"schema_version", 1},
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

int main(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(benchmarkOptions);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Generate and evaluate structured quantum benchmarks\n");

  try {
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
      const auto source = instanceSpecificationPath == "-"
                              ? "<stdin>"
                              : instanceSpecificationPath.getValue();
      return generateFromInstanceSpecification(instanceSpecification, source);
    }
    if (evaluateCommand) {
      if (manifestInputPath == "-") {
        throw std::invalid_argument("--manifest requires a file path");
      }
      const auto manifest = readText(manifestInputPath);
      const auto counts = readText(countsInputPath);
      const auto countsSource =
          countsInputPath == "-" ? "<stdin>" : countsInputPath.getValue();
      llvm::outs() << mqt::bench::evaluateJSON(manifest, counts,
                                               manifestInputPath, countsSource)
                   << '\n';
      return 0;
    }
    llvm::errs() << "a command is required; use --help for usage\n";
  } catch (const std::exception& exception) {
    llvm::errs() << exception.what() << '\n';
  }
  return 1;
}
