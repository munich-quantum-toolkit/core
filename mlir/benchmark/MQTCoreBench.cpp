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
 * @brief Generates and evaluates structured benchmark instances.
 */

#include "benchmarks/JSON.hpp"
#include "mlir/Benchmark/Generate.h"
#include "mlir/Compiler/Programs.h"

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
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

namespace {

llvm::cl::OptionCategory benchmarkOptions("Benchmark options");
llvm::cl::SubCommand listCommand("list", "List the available benchmarks");
llvm::cl::SubCommand describeCommand("describe",
                                     "Describe one benchmark request schema");
llvm::cl::SubCommand generateCommand("generate",
                                     "Generate one configured benchmark");
llvm::cl::SubCommand evaluateCommand("evaluate",
                                     "Evaluate counts against a manifest");

llvm::cl::opt<std::string> benchmarkId(llvm::cl::Positional,
                                       llvm::cl::desc("<id>"),
                                       llvm::cl::Required,
                                       llvm::cl::cat(benchmarkOptions),
                                       llvm::cl::sub(describeCommand));

llvm::cl::opt<std::string> requestPath(
    "request", llvm::cl::desc("Request JSON file, or '-' for standard input"),
    llvm::cl::value_desc("file|-"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(generateCommand));
llvm::cl::opt<std::string> outputFormat(
    "format", llvm::cl::desc("Generated program format: qc or jeff"),
    llvm::cl::value_desc("qc|jeff"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(generateCommand));
llvm::cl::opt<std::string> outputDirectory(
    "output", llvm::cl::desc("Directory for the program and manifest"),
    llvm::cl::value_desc("directory"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(generateCommand));
llvm::cl::opt<bool>
    overwrite("overwrite", llvm::cl::desc("Replace existing generated files"),
              llvm::cl::init(false), llvm::cl::cat(benchmarkOptions),
              llvm::cl::sub(generateCommand));

llvm::cl::opt<std::string> manifestInputPath(
    "manifest", llvm::cl::desc("Benchmark manifest JSON file"),
    llvm::cl::value_desc("file"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(evaluateCommand));
llvm::cl::opt<std::string> countsInputPath(
    "counts", llvm::cl::desc("Counts JSON file, or '-' for standard input"),
    llvm::cl::value_desc("file|-"), llvm::cl::Required,
    llvm::cl::cat(benchmarkOptions), llvm::cl::sub(evaluateCommand));

[[nodiscard]] std::string readText(const std::string& path) {
  auto buffer = path == "-" ? llvm::MemoryBuffer::getSTDIN()
                            : llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    throw std::runtime_error("failed to read '" + path +
                             "': " + buffer.getError().message());
  }
  return (*buffer)->getBuffer().str();
}

[[nodiscard]] bool pathExists(const std::filesystem::path& path) {
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

bool validateOutputTarget(const std::filesystem::path& path,
                          const bool mayOverwrite) {
  if (!pathExists(path)) {
    return false;
  }
  if (!mayOverwrite) {
    throw std::runtime_error("refusing to overwrite existing file '" +
                             path.string() + "'");
  }

  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || status.type() != std::filesystem::file_type::regular) {
    throw std::runtime_error("refusing to overwrite non-regular file '" +
                             path.string() + "'");
  }
  return true;
}

struct OpenSibling {
  std::filesystem::path path;
  int descriptor;
};

[[nodiscard]] std::filesystem::path
siblingPath(const std::filesystem::path& finalPath,
            const std::string_view purpose) {
  const auto random =
      (static_cast<uint64_t>(llvm::sys::Process::GetRandomNumber()) << 32U) |
      llvm::sys::Process::GetRandomNumber();
  const auto name = finalPath.filename().string() + "." + std::string(purpose) +
                    "-" + llvm::utohexstr(random);
  return finalPath.parent_path() / name;
}

[[nodiscard]] OpenSibling createSibling(const std::filesystem::path& finalPath,
                                        const std::string_view purpose) {
  for (size_t attempt = 0; attempt < 32; ++attempt) {
    auto path = siblingPath(finalPath, purpose);
    int descriptor = -1;
    const auto error = llvm::sys::fs::openFileForWrite(
        path.string(), descriptor, llvm::sys::fs::CD_CreateNew);
    if (!error) {
      return {std::move(path), descriptor};
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

[[nodiscard]] std::filesystem::path
stageFile(const std::filesystem::path& finalPath,
          const std::string_view contents) {
  auto temporary = createSibling(finalPath, "tmp");
  llvm::raw_fd_ostream stream(temporary.descriptor, true);
  stream.write(contents.data(), contents.size());
  stream.close();
  const auto error = stream.error();
  stream.clear_error();
  if (error) {
    llvm::sys::fs::remove(temporary.path.string());
    throw std::runtime_error("failed to write temporary output for '" +
                             finalPath.string() + "': " + error.message());
  }
  return temporary.path;
}

[[nodiscard]] std::optional<std::filesystem::path>
backupFile(const std::filesystem::path& finalPath, const bool exists) {
  if (!exists) {
    return std::nullopt;
  }

  for (size_t attempt = 0; attempt < 32; ++attempt) {
    auto path = siblingPath(finalPath, "backup");
    const auto error =
        llvm::sys::fs::create_hard_link(finalPath.string(), path.string());
    if (!error) {
      return path;
    }
    if (error != std::errc::file_exists) {
      throw std::runtime_error("failed to back up '" + finalPath.string() +
                               "': " + error.message());
    }
  }
  throw std::runtime_error("failed to choose a unique backup file next to '" +
                           finalPath.string() + "'");
}

void removeIfPresent(const std::optional<std::filesystem::path>& path) {
  if (path) {
    llvm::sys::fs::remove(path->string());
  }
}

void restoreFile(const std::filesystem::path& finalPath,
                 std::optional<std::filesystem::path>& backupPath) {
  const auto error = backupPath ? llvm::sys::fs::rename(backupPath->string(),
                                                        finalPath.string())
                                : llvm::sys::fs::remove(finalPath.string());
  if (error) {
    llvm::errs() << "failed to restore '" << finalPath.string()
                 << "' after an output error: " << error.message();
    if (backupPath) {
      llvm::errs() << "; recovery backup remains at '" << backupPath->string()
                   << "'";
      backupPath.reset();
    }
    llvm::errs() << '\n';
  }
}

void removeOwnedOutput(const std::filesystem::path& temporaryPath,
                       const std::filesystem::path& finalPath) {
  std::error_code error;
  if (std::filesystem::equivalent(temporaryPath, finalPath, error) && !error) {
    llvm::sys::fs::remove(finalPath.string());
  }
}

[[nodiscard]] std::string programExtension(const std::string_view format) {
  if (format == "qc") {
    return ".qc.mlir";
  }
  if (format == "jeff") {
    return ".jeff";
  }
  throw std::invalid_argument("unsupported output format '" +
                              std::string(format) + "'");
}

template <class Benchmark>
[[nodiscard]] int
generate(const std::string_view id, const Benchmark& benchmark,
         const std::string_view format, const std::filesystem::path& directory,
         const bool mayOverwrite) {
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

  const auto caseId = mqt::benchmarks::caseId(benchmark);
  const auto baseName = std::string(id) + "-" + caseId;
  const auto programPath = directory / (baseName + extension);
  const auto manifestPath =
      directory / (baseName + "." + std::string(format) + ".manifest.json");
  const auto programExisted = validateOutputTarget(programPath, mayOverwrite);
  const auto manifestExisted = validateOutputTarget(manifestPath, mayOverwrite);

  auto program = mqt::benchmark::generateProgram(benchmark);
  if (!program) {
    return 1;
  }

  std::string serializedProgram;
  if (format == "qc") {
    serializedProgram = program->str();
    if (serializedProgram.empty() || serializedProgram.back() != '\n') {
      serializedProgram.push_back('\n');
    }
  } else {
    auto compiled = mlir::runDefaultPipeline(std::move(*program),
                                             mlir::ProgramFormat::Jeff);
    if (!compiled) {
      llvm::errs() << id << ": failed to build the jeff program\n";
      return 1;
    }
    const auto bytes = std::get<mlir::JeffProgram>(*compiled).toBytes();
    serializedProgram.assign(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size());
  }
  auto manifest = mqt::benchmarks::toManifestJSON(benchmark);
  manifest.push_back('\n');

  std::optional<std::filesystem::path> temporaryProgram;
  std::optional<std::filesystem::path> temporaryManifest;
  std::optional<std::filesystem::path> programBackup;
  std::optional<std::filesystem::path> manifestBackup;
  const auto removeTemporaryAndBackups = llvm::make_scope_exit([&] {
    removeIfPresent(temporaryProgram);
    removeIfPresent(temporaryManifest);
    removeIfPresent(programBackup);
    removeIfPresent(manifestBackup);
  });

  temporaryProgram = stageFile(programPath, serializedProgram);
  temporaryManifest = stageFile(manifestPath, manifest);

  if (!mayOverwrite) {
    if (const auto linkError = llvm::sys::fs::create_hard_link(
            temporaryProgram->string(), programPath.string())) {
      llvm::errs() << "failed to publish '" << programPath.string()
                   << "': " << linkError.message() << '\n';
      return 1;
    }
    if (const auto linkError = llvm::sys::fs::create_hard_link(
            temporaryManifest->string(), manifestPath.string())) {
      removeOwnedOutput(*temporaryProgram, programPath);
      llvm::errs() << "failed to publish '" << manifestPath.string()
                   << "': " << linkError.message() << '\n';
      return 1;
    }
    removeIfPresent(temporaryProgram);
    removeIfPresent(temporaryManifest);
    temporaryProgram.reset();
    temporaryManifest.reset();
  } else {
    programBackup = backupFile(programPath, programExisted);
    manifestBackup = backupFile(manifestPath, manifestExisted);

    if (const auto renameError = llvm::sys::fs::rename(
            temporaryProgram->string(), programPath.string())) {
      if (programExisted) {
        restoreFile(programPath, programBackup);
      }
      llvm::errs() << "failed to publish '" << programPath.string()
                   << "': " << renameError.message() << '\n';
      return 1;
    }
    temporaryProgram.reset();

    if (const auto renameError = llvm::sys::fs::rename(
            temporaryManifest->string(), manifestPath.string())) {
      restoreFile(programPath, programBackup);
      restoreFile(manifestPath, manifestBackup);
      llvm::errs() << "failed to publish '" << manifestPath.string()
                   << "': " << renameError.message() << '\n';
      return 1;
    }
    temporaryManifest.reset();
  }

  llvm::json::Object response{{"benchmark", std::string(id)},
                              {"case_id", caseId},
                              {"format", std::string(format)},
                              {"manifest_path", manifestPath.string()},
                              {"program_path", programPath.string()},
                              {"schema_version", 1}};
  llvm::outs() << llvm::json::Value(std::move(response)) << '\n';
  return 0;
}

[[nodiscard]] int generateRequest(const std::string& request,
                                  const std::string& source) {
  const auto id = mqt::benchmarks::benchmarkIdFromRequestJSON(request, source);
  const auto directory = std::filesystem::path(outputDirectory.getValue());
  if (id == "ghz") {
    return generate(id, mqt::benchmarks::ghzFromRequestJSON(request, source),
                    outputFormat, directory, overwrite);
  }
  if (id == "grover") {
    return generate(id, mqt::benchmarks::groverFromRequestJSON(request, source),
                    outputFormat, directory, overwrite);
  }
  if (id == "qpe") {
    return generate(id, mqt::benchmarks::qpeFromRequestJSON(request, source),
                    outputFormat, directory, overwrite);
  }
  throw std::invalid_argument("unsupported benchmark '" + id + "'");
}

template <class Benchmark>
[[nodiscard]] std::string evaluate(const Benchmark& benchmark,
                                   const mqt::benchmarks::Counts& counts) {
  const auto shots = std::accumulate(
      counts.begin(), counts.end(), size_t{0},
      [](const size_t sum, const auto& item) { return sum + item.second; });
  return mqt::benchmarks::evaluationToJSON(mqt::benchmarks::caseId(benchmark),
                                           shots, benchmark.evaluate(counts));
}

[[nodiscard]] std::string evaluateCounts(const std::string& manifest,
                                         const std::string& manifestSource,
                                         const std::string& counts,
                                         const std::string& countsSource) {
  const auto id =
      mqt::benchmarks::benchmarkIdFromManifestJSON(manifest, manifestSource);
  const auto parsedCounts =
      mqt::benchmarks::countsFromJSON(counts, countsSource);
  if (id == "ghz") {
    return evaluate(
        mqt::benchmarks::ghzFromManifestJSON(manifest, manifestSource),
        parsedCounts);
  }
  if (id == "grover") {
    return evaluate(
        mqt::benchmarks::groverFromManifestJSON(manifest, manifestSource),
        parsedCounts);
  }
  if (id == "qpe") {
    return evaluate(
        mqt::benchmarks::qpeFromManifestJSON(manifest, manifestSource),
        parsedCounts);
  }
  throw std::invalid_argument("unsupported benchmark '" + id + "'");
}

} // namespace

int main(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(benchmarkOptions);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Generate and evaluate structured quantum benchmarks\n");

  try {
    if (listCommand) {
      llvm::outs() << mqt::benchmarks::listBenchmarksJSON() << '\n';
      return 0;
    }
    if (describeCommand) {
      llvm::outs() << mqt::benchmarks::describeBenchmarkJSON(benchmarkId)
                   << '\n';
      return 0;
    }
    if (generateCommand) {
      const auto request = readText(requestPath);
      const auto source =
          requestPath == "-" ? "<stdin>" : requestPath.getValue();
      return generateRequest(request, source);
    }
    if (evaluateCommand) {
      if (manifestInputPath == "-") {
        throw std::invalid_argument("--manifest requires a file path");
      }
      const auto manifest = readText(manifestInputPath);
      const auto counts = readText(countsInputPath);
      const auto countsSource =
          countsInputPath == "-" ? "<stdin>" : countsInputPath.getValue();
      llvm::outs() << evaluateCounts(manifest, manifestInputPath, counts,
                                     countsSource)
                   << '\n';
      return 0;
    }
    llvm::errs() << "a command is required; use --help for usage\n";
  } catch (const std::exception& exception) {
    llvm::errs() << exception.what() << '\n';
  }
  return 1;
}
