/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"
#include "mlir/Compiler/TargetEnvironment.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using namespace std::filesystem;

namespace {
llvm::cl::opt<std::string>
    mlirDir("mlir-dir",
            llvm::cl::desc("Directory containing .mlir files to benchmark"),
            llvm::cl::value_desc("directory path"));
}

struct BenchmarkEntry {
  std::string name;
  std::function<OwningOpRef<ModuleOp>(MLIRContext*)> fn;
};

/// Statistics structure to collect statistics of one pass run.
struct BenchmarkResult {
  size_t numRoutingSWAPs{0};
  size_t numAppendixSWAPs{0};
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;

  /// Return the ellapsed time in milliseconds.
  [[nodiscard]] size_t ellapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
  }
};

struct AggregateBenchmarkResult {
  std::string name;
  SmallVector<BenchmarkResult> stats;

  [[nodiscard]] size_t minTime() const {
    const auto* it = llvm::min_element(
        stats, [](const BenchmarkResult& a, const BenchmarkResult& b) {
          return a.ellapsed() < b.ellapsed();
        });
    if (it != stats.end()) {
      return it->ellapsed();
    }

    return std::numeric_limits<size_t>::max();
  }

  [[nodiscard]] size_t maxTime() const {
    const auto* it = llvm::max_element(
        stats, [](const BenchmarkResult& a, const BenchmarkResult& b) {
          return a.ellapsed() < b.ellapsed();
        });
    if (it != stats.end()) {
      return it->ellapsed();
    }

    return std::numeric_limits<size_t>::max();
  }

  [[nodiscard]] size_t avgTime() const {
    size_t sum = 0;
    for (const auto& s : stats) {
      sum += s.ellapsed();
    }
    return sum / stats.size();
  }

  [[nodiscard]] size_t minNumRoutingSWAPs() const {
    const auto* it = llvm::min_element(
        stats, [](const BenchmarkResult& a, const BenchmarkResult& b) {
          return a.numRoutingSWAPs < b.numRoutingSWAPs;
        });
    if (it != stats.end()) {
      return it->numRoutingSWAPs;
    }

    return std::numeric_limits<size_t>::max();
  }

  [[nodiscard]] size_t maxNumRoutingSWAPs() const {
    const auto* it = llvm::max_element(
        stats, [](const BenchmarkResult& a, const BenchmarkResult& b) {
          return a.numRoutingSWAPs < b.numRoutingSWAPs;
        });
    if (it != stats.end()) {
      return it->numRoutingSWAPs;
    }

    return std::numeric_limits<size_t>::max();
  }

  [[nodiscard]] size_t avgNumRoutingSWAPs() const {
    size_t sum = 0;
    for (const auto& s : stats) {
      sum += s.numRoutingSWAPs;
    }
    return sum / stats.size();
  }

  [[nodiscard]] size_t minNumAppendixSWAPs() const {
    const auto* it = llvm::min_element(
        stats, [](const BenchmarkResult& a, const BenchmarkResult& b) {
          return a.numAppendixSWAPs < b.numAppendixSWAPs;
        });
    if (it != stats.end()) {
      return it->numAppendixSWAPs;
    }

    return std::numeric_limits<size_t>::max();
  }

  [[nodiscard]] size_t maxNumAppendixSWAPs() const {
    const auto* it = llvm::max_element(
        stats, [](const BenchmarkResult& a, const BenchmarkResult& b) {
          return a.numAppendixSWAPs < b.numAppendixSWAPs;
        });
    if (it != stats.end()) {
      return it->numAppendixSWAPs;
    }

    return std::numeric_limits<size_t>::max();
  }

  [[nodiscard]] size_t avgNumAppendixSWAPs() const {
    size_t sum = 0;
    for (const auto& s : stats) {
      sum += s.numAppendixSWAPs;
    }
    return sum / stats.size();
  }
};

/// Custom pass instrumentation to collect timing and statistics.
class BenchmarkPassInstrumentation : public PassInstrumentation {
public:
  explicit BenchmarkPassInstrumentation(BenchmarkResult& stats)
      : stats(&stats) {}

  void runBeforePass([[maybe_unused]] Pass* pass, Operation* op) override {
    stats->start = std::chrono::high_resolution_clock::now();
  }

  void runAfterPass([[maybe_unused]] Pass* pass,
                    [[maybe_unused]] Operation* op) override {
    stats->end = std::chrono::high_resolution_clock::now();
    for (const auto& s : pass->getStatistics()) {
      if (s->getName() == "num-routing-swaps") {
        stats->numRoutingSWAPs = s->getValue();
      }
      if (s->getName() == "num-appendix-swaps") {
        stats->numAppendixSWAPs = s->getValue();
      }
    }
  }

private:
  BenchmarkResult* stats;
};

/// Return a n x m square-grid compiler target.
static CompilerTarget getSquareGridTarget(size_t rows, size_t cols) {
  const auto numSites = rows * cols;

  std::vector<CompilerTarget::Coupling> couplings;
  couplings.reserve(numSites * 2);

  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      const auto i = (r * cols) + c;
      // Horizontal couplings
      if (c + 1 < cols) {
        couplings.emplace_back(i, i + 1);
      }
      // Vertical couplings
      if (r + 1 < rows) {
        couplings.emplace_back(i, i + cols);
      }
    }
  }

  return llvm::cantFail(CompilerTarget::create(
      numSites, CompilerTarget::Connectivity::fromCouplings(couplings),
      CompilerTarget::NativeOperations::fromOperations({})));
}

/// Run the mapping pass and collect timing statistics.
static AggregateBenchmarkResult runBenchmark(MLIRContext* context,
                                             const BenchmarkEntry& entry,
                                             const CompilerTarget& target,
                                             const size_t numRepeats = 10) {

  static const auto PAYLOAD = [] {
    PayloadFormat format;
    format.id = "test.payload";
    format.version = "1.0.0";
    return llvm::cantFail(PayloadSpecification::create(std::move(format)));
  }();

  AggregateBenchmarkResult aggStats;
  aggStats.name = entry.name;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> seedDist;

  for (size_t r = 0; r < numRepeats; ++r) {
    BenchmarkResult stats;
    BenchmarkPassInstrumentation instrumentation(stats);
    MappingPassOptions options{.nlookahead = 20,
                               .lambda = 0.5,
                               .niterations = 1,
                               .ntrials = 1,
                               .seed = seedDist(gen)};

    PassManager pm(context);
    pm.addInstrumentation(
        std::make_unique<BenchmarkPassInstrumentation>(instrumentation));
    pm.addPass(createMappingPass(options));

    auto mod = entry.fn(context);
    attachTargetEnvironment(*mod, TargetEnvironment(target, PAYLOAD));
    if (failed(pm.run(*mod))) {
      llvm::errs() << "Pass failed for circuit: " << entry.name << "\n";
      continue;
    }
    aggStats.stats.emplace_back(stats);
  }

  return aggStats;
}

/// Print benchmark results as in comma-seperated format.
static void printCSV(const SmallVector<AggregateBenchmarkResult>& allStats,
                     const char sep = ';') {
  llvm::outs() << "Circuit Name" << sep << "Min Time" << sep << "Max Time"
               << sep << "Avg Time" << sep << "Min Routing SWAPs" << sep
               << "Max Routing SWAPs" << sep << "Avg Routing SWAPs" << sep
               << "Min Appendix SWAPs" << sep << "Max Appendix SWAPs" << sep
               << "Avg Appendix SWAPs" << sep << "\n";

  // Print each benchmark's stats
  for (const auto& stats : allStats) {
    llvm::outs() << stats.name << sep << stats.minTime() << sep
                 << stats.maxTime() << sep << stats.avgTime() << sep
                 << stats.minNumRoutingSWAPs() << sep
                 << stats.maxNumRoutingSWAPs() << sep
                 << stats.avgNumRoutingSWAPs() << sep
                 << stats.minNumAppendixSWAPs() << sep
                 << stats.maxNumAppendixSWAPs() << sep
                 << stats.avgNumAppendixSWAPs() << "\n";
  }
}

/// Parse an MLIR file and return the module
static OwningOpRef<ModuleOp> parseMLIRFile(MLIRContext* context,
                                           const path& filePath) {
  llvm::SourceMgr sourceMgr;

  // Try to open the file using LLVM's file system
  auto file = llvm::MemoryBuffer::getFile(filePath.string());
  if (!file) {
    llvm::errs() << "Error: Could not open MLIR file: " << filePath.string()
                 << ": " << file.getError().message() << "\n";
    return nullptr;
  }

  sourceMgr.AddNewSourceBuffer(std::move(*file), llvm::SMLoc());

  // Parse the MLIR
  auto mod = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!mod) {
    llvm::errs() << "Error: Failed to parse MLIR file: " << filePath.string()
                 << "\n";
    return nullptr;
  }

  return mod;
}

/// Load all .mlir files from a directory and create benchmark entries
static SmallVector<BenchmarkEntry>
loadMLIRBenchmarks(MLIRContext* context, const std::string& directory) {

  if (directory.empty()) {
    llvm::errs()
        << "Error: No MLIR directory specified. Use --mlir-dir option.\n";
    return {};
  }

  std::filesystem::path dirPath(directory);
  if (!exists(dirPath) || !is_directory(dirPath)) {
    llvm::errs()
        << "Error: MLIR directory does not exist or is not a directory: "
        << directory << "\n";
    return {};
  }

  // Iterate through all .mlir files in the directory
  SmallVector<BenchmarkEntry> entries;
  for (const auto& entry : directory_iterator(dirPath)) {
    if (entry.is_regular_file() && entry.path().extension() == ".mlir") {
      std::string name = entry.path().stem().string(); // Remove .mlir extension

      // Create benchmark entry
      entries.emplace_back(name, [filePath = entry.path()](MLIRContext* ctx) {
        return parseMLIRFile(ctx, filePath);
      });
    }
  }

  return entries;
}

int main(int argc, char** argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MQT Core Mapping Benchmark\n");

  CompilerTarget target = getSquareGridTarget(10, 12);

  MLIRContext context;
  DialectRegistry registry;
  registry.insert<QCODialect, qtensor::QTensorDialect, scf::SCFDialect,
                  mqt::MQTDialect, cbit::CBitDialect, arith::ArithDialect,
                  func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Load benchmark entries from MLIR files
  SmallVector<BenchmarkEntry> entries = loadMLIRBenchmarks(&context, mlirDir);

  if (entries.empty()) {
    llvm::errs() << "Error: No valid .mlir files found in directory: "
                 << mlirDir << "\n";
    return 1;
  }

  SmallVector<AggregateBenchmarkResult> allAggStats;
  for (const auto& entry : entries) {
    llvm::dbgs() << "[benchmark] " << entry.name << " qubits!\n";
    allAggStats.emplace_back(runBenchmark(&context, entry, target, 5));
  }

  printCSV(allAggStats);
  return 0;
}
