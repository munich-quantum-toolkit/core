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
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::qco;

struct BenchmarkEntry {
  std::string name;
  size_t numQubits;
  std::function<OwningOpRef<ModuleOp>(MLIRContext*)> fn;
};

/// Statistics structure to collect statistics of one pass run.
struct BenchmarkResult {
  size_t numUnitaries{0};
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
  size_t numQubits{0};
  SmallVector<BenchmarkResult> stats;

  [[nodiscard]] size_t numUnitaries() const {
    return stats.front().numUnitaries;
  }

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
    op->walk([&](UnitaryOpInterface) { ++stats->numUnitaries; });
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

  return llvm::cantFail(CompilerTarget::create(numSites, std::move(couplings)));
}

/// Return a structured program implementing Grover's algorithm using
/// @p numQubits qubits.
static OwningOpRef<ModuleOp> groverAlg(MLIRContext* context,
                                       const int64_t nqubits,
                                       const int64_t niterations,
                                       const std::string& markedBitstring) {
  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>(nqubits, builder.getI1Type()));

  SmallVector<Value> qubits(nqubits);
  SmallVector<Value> bits(nqubits);

  Value tensor = builder.qtensorAlloc(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  // Initialize: apply Hadamard to all qubits to create uniform superposition
  for (int64_t i = 0; i < nqubits; ++i) {
    qubits[i] = builder.h(qubits[i]);
  }

  // Grover iterations using scf.For
  qubits =
      builder.scfFor(1, niterations, 1, qubits, [&](Value, ValueRange args) {
        SmallVector<Value> bodyQubits(args);
        ArrayRef bodyRef(bodyQubits);

        for (size_t i = 0; i < nqubits; ++i) {
          if (markedBitstring[nqubits - 1 - i] == '0') {
            bodyQubits[i] = builder.x(bodyQubits[i]);
          }
        }

        auto mczOut = builder.mcz(bodyRef.drop_back(), bodyRef.back());
        for (size_t i = 0; i < nqubits - 1; ++i) {
          bodyQubits[i] = mczOut.first[i];
        }
        bodyQubits[nqubits - 1] = mczOut.second;

        for (size_t i = 0; i < nqubits; ++i) {
          if (markedBitstring[nqubits - 1 - i] == '0') {
            bodyQubits[i] = builder.x(bodyQubits[i]);
          }
        }

        //
        // Diffusion operator (2|00..0><00.0| - I)
        //

        for_each(bodyQubits, [&](auto& q) { q = builder.h(q); });
        for_each(bodyQubits, [&](auto& q) { q = builder.x(q); });

        mczOut = builder.mcz(bodyRef.drop_back(), bodyRef.back());
        for (size_t i = 0; i < nqubits - 1; ++i) {
          bodyQubits[i] = mczOut.first[i];
        }
        bodyQubits[nqubits - 1] = mczOut.second;

        for_each(bodyQubits, [&](auto& q) { q = builder.x(q); });
        for_each(bodyQubits, [&](auto& q) { q = builder.h(q); });

        return bodyQubits;
      });

  qubits = builder.barrier(qubits);

  // Measure all qubits
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  // Clean up
  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto mod = builder.finalize(bits);

  PassManager pm(context);
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);

  return mod;
}

static OwningOpRef<ModuleOp> qaoa(MLIRContext* context, const int64_t nqubits,
                                  const int64_t nlayers, const double gamma,
                                  const double beta) {
  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>(nqubits, builder.getI1Type()));

  SmallVector<Value> qubits(nqubits);
  SmallVector<Value> bits(nqubits);

  Value tensor = builder.qtensorAlloc(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  // Initialize: apply Hadamard to all qubits to create uniform superposition
  for_each(qubits, [&](auto& q) { q = builder.h(q); });

  // Each layer applies the cost operator of a ring of couplings and then the
  // mixer. The problem graph is fixed, so the layer count is a constant.

  qubits = builder.scfFor(0, nlayers, 1, qubits, [&](Value, ValueRange args) {
    SmallVector<Value> bodyQubits(args);

    for (int64_t i = 0; i < nqubits - 1; ++i) {
      std::tie(bodyQubits[i], bodyQubits[i + 1]) =
          builder.rzz(gamma, bodyQubits[i], bodyQubits[i + 1]);
    }

    std::tie(bodyQubits[nqubits - 1], bodyQubits[0]) =
        builder.rzz(beta, bodyQubits[nqubits - 1], bodyQubits[0]);

    for_each(bodyQubits, [&](auto& q) { q = builder.rx(beta, q); });

    return bodyQubits;
  });

  qubits = builder.barrier(qubits);

  // Measure all qubits
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  // Clean up
  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto mod = builder.finalize(bits);

  PassManager pm(context);
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);

  return mod;
}

static OwningOpRef<ModuleOp>
groverDecomposed(MLIRContext* context, const int64_t nqubits,
                 const int64_t niterations,
                 const std::string& markedBitstring) {
  auto mod = groverAlg(context, nqubits, niterations, markedBitstring);
  PassManager pm(context);
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);
  return mod;
}

/// Return a straight-line program implementing Grover's algorithm using
/// @p numQubits qubits with |11..1> as the target state.
static OwningOpRef<ModuleOp>
groverUnrolled(MLIRContext* context, const int64_t nqubits,
               const int64_t niterations, const std::string& markedBitstring) {
  auto mod = groverAlg(context, nqubits, niterations, markedBitstring);
  PassManager pm(context);
  pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);
  return mod;
}

static OwningOpRef<ModuleOp> magicState(MLIRContext* context,
                                        const int64_t ncopies,
                                        const int64_t nstabilizers) {

  // Generate random stabilizer generators: 'X', 'Z', or '.' (identity)
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 2);

  // Create a 2D vector for generators
  std::vector<std::vector<char>> generators(nstabilizers,
                                            std::vector<char>(ncopies));

  // Possible Pauli operators
  const char paulis[] = {'X', 'Z', '.'};

  // Fill with random Pauli operators
  for (int64_t s = 0; s < nstabilizers; ++s) {
    for (int64_t j = 0; j < ncopies; ++j) {
      generators[s][j] = paulis[dis(gen)];
    }
  }

  const int64_t nqubits = ncopies + 1; // MAGIC_COPIES + ancilla

  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>(nqubits, builder.getI1Type()));

  SmallVector<Value> qubits(nqubits);
  SmallVector<Value> bits(nqubits);

  Value tensor = builder.qtensorAlloc(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  // A round prepares five noisy copies and reads the four stabilizers of the
  // five-qubit code. It is accepted only when every syndrome bit is trivial,
  // and a rejected round throws the copies away and starts over, so the number
  // of rounds depends on the measurements.
  auto syndrome = builder.allocClassicalBitRegister(nstabilizers, "syndrome");
  qubits = builder.scfWhile(
      qubits,
      [&](ValueRange args) {
        SmallVector<Value> bodyQubits(args);
        auto& anc = bodyQubits[bodyQubits.size() - 1];

        for (int64_t i = 0; i < ncopies; ++i) {
          bodyQubits[i] = builder.h(bodyQubits[i]);
          bodyQubits[i] = builder.t(bodyQubits[i]);
        }

        for (int64_t s = 0; s < nstabilizers; ++s) {
          anc = builder.reset(anc);
          anc = builder.h(anc);

          for (int64_t j = 0; j < ncopies; ++j) {
            if (generators[s][j] == 'X') {
              std::tie(anc, bodyQubits[j]) = builder.cx(anc, bodyQubits[j]);
            } else if (generators[s][j] == 'Z') {
              std::tie(anc, bodyQubits[j]) = builder.cz(anc, bodyQubits[j]);
            }
          }
          anc = builder.h(anc);
          std::tie(anc, std::ignore) = builder.measure(anc, syndrome, s);
        }

        auto rejected = builder.loadClassicalBit(syndrome, 0);
        for (int64_t s = 1; s < nstabilizers; ++s) {
          rejected = arith::OrIOp::create(
              builder, rejected, builder.loadClassicalBit(syndrome, s));
        }

        builder.scfCondition(rejected, bodyQubits);
        return bodyQubits;
      },
      [&](ValueRange args) { return llvm::to_vector(args); });

  qubits = builder.barrier(qubits);

  // Measure all qubits
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  // Clean up
  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  auto mod = builder.finalize(bits);

  PassManager pm(context);
  pm.addPass(createDecomposeMultiControlled());
  pm.addPass(createCanonicalizerPass());
  pm.run(*mod);

  return mod;
}

static OwningOpRef<ModuleOp> vqe(MLIRContext* context, const int64_t nqubits,
                                 const int64_t nlayers, const double decay) {
  /// The angle the optimizer starts from.
  constexpr double vqeInitialAngle = llvm::numbers::pi / 2.0;

  QCOProgramBuilder builder(context);
  builder.initialize(SmallVector<Type>{builder.getF64Type()});

  auto reg = builder.allocClassicalBitRegister(nqubits, "reg");

  SmallVector<Value> qubits(nqubits);

  auto tensor = builder.qtensorAlloc(nqubits);
  for (int64_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) = builder.qtensorExtract(tensor, i);
  }

  SmallVector<Value> whileArgs{builder.floatConstant(vqeInitialAngle),
                               // The chain has at most `nqubits - 1`
                               // disagreeing pairs, so the first round
                               // improves on this value whatever it measures.
                               builder.intConstant(nqubits)};
  whileArgs.append(qubits);

  // A round prepares the ansatz at the current angle, reads the register, and
  // estimates the energy of an Ising chain from the measured bits. The
  // optimizer shrinks the angle and runs another round only while the energy
  // improves on the round before it, so the rounds follow from the
  // measurements.

  whileArgs = builder.scfWhile(
      whileArgs,
      [&](ValueRange args) {
        SmallVector<Value> bodyArgs(args);

        auto& angle = bodyArgs[0];
        auto& previous = bodyArgs[1];
        SmallVector<Value> whileBodyQubits(ArrayRef(bodyArgs).drop_front(2));

        for_each(whileBodyQubits, [&](auto& q) { q = builder.reset(q); });

        whileBodyQubits = builder.scfFor(
            0, nlayers, 1, whileBodyQubits, [&](Value, ValueRange forArgs) {
              SmallVector<Value> forBodyQubits(forArgs);
              for_each(forBodyQubits,
                       [&](auto& q) { q = builder.ry(angle, q); });

              for (size_t i = 0; i < nqubits - 1; ++i) {
                std::tie(forBodyQubits[i], forBodyQubits[i + 1]) =
                    builder.cx(forBodyQubits[i], forBodyQubits[i + 1]);
              }
              return forBodyQubits;
            });

        whileBodyQubits = builder.barrier(whileBodyQubits);

        for (int64_t i = 0; i < whileBodyQubits.size(); ++i) {
          std::tie(whileBodyQubits[i], std::ignore) =
              builder.measure(whileBodyQubits[i], reg, i);
        }

        for (size_t i = 2; i < bodyArgs.size(); ++i) {
          bodyArgs[i] = whileBodyQubits[i - 2];
        }

        // auto energy =
        //     func::CallOp::create(builder, "getEnergy",
        //                          SmallVector<Type>{builder.getI64Type()}, reg)
        //         .getResult(0);

        // auto improved = arith::CmpIOp::create(
        //     builder, arith::CmpIPredicate::slt, energy, previous);

        auto improved = builder.boolConstant(true);

        builder.scfCondition(improved, bodyArgs);
        return bodyArgs;
      },
      [&](ValueRange args) {
        SmallVector<Value> bodyArgs(args);
        auto decayValue = builder.floatConstant(decay);
        bodyArgs[0] =
            arith::MulFOp::create(builder, bodyArgs[0], decayValue).getResult();
        return bodyArgs;
      });

  qubits = to_vector(ArrayRef(whileArgs).drop_front(2));

  qubits = builder.barrier(qubits);

  for (int64_t i = 0; i < nqubits; ++i) {
    tensor = builder.qtensorInsert(qubits[i], tensor, i);
  }

  builder.qtensorDealloc(tensor);

  return builder.finalize(whileArgs[0]);
  ;
}

/// Run the mapping pass and collect timing statistics.
static AggregateBenchmarkResult runBenchmark(MLIRContext* context,
                                             const BenchmarkEntry& entry,
                                             const CompilerTarget& target,
                                             const size_t numRepeats = 10) {

  AggregateBenchmarkResult aggStats;
  aggStats.name = entry.name;
  aggStats.numQubits = entry.numQubits;

  for (size_t r = 0; r < numRepeats; ++r) {
    BenchmarkResult stats;
    BenchmarkPassInstrumentation instrumentation(stats);
    MappingPassOptions options{
        .nlookahead = 20, .lambda = 0.5, .niterations = 1, .ntrials = 18};

    PassManager pm(context);
    pm.addInstrumentation(
        std::make_unique<BenchmarkPassInstrumentation>(instrumentation));
    pm.addPass(createMappingPass(target, options));

    auto mod = entry.fn(context);
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
  llvm::outs() << "Circuit Name" << sep << "NumQubits" << sep << "NumUnitaries"
               << sep << "Min Time" << sep << "Max Time" << sep << "Avg Time"
               << sep << "Min Routing SWAPs" << sep << "Max Routing SWAPs"
               << sep << "Avg Routing SWAPs" << sep << "Min Appendix SWAPs"
               << sep << "Max Appendix SWAPs" << sep << "Avg Appendix SWAPs"
               << sep << "\n";

  // Print each benchmark's stats
  for (const auto& stats : allStats) {
    llvm::outs() << stats.name << sep << stats.numQubits << sep
                 << stats.numUnitaries() << sep << stats.minTime() << sep
                 << stats.maxTime() << sep << stats.avgTime() << sep
                 << stats.minNumRoutingSWAPs() << sep
                 << stats.maxNumRoutingSWAPs() << sep
                 << stats.avgNumRoutingSWAPs() << sep
                 << stats.minNumAppendixSWAPs() << sep
                 << stats.maxNumAppendixSWAPs() << sep
                 << stats.avgNumAppendixSWAPs() << "\n";
  }
}

int main(int argc, char** argv) {
  CompilerTarget target = getSquareGridTarget(10, 12);

  MLIRContext context;
  DialectRegistry registry;
  registry.insert<QCODialect, qtensor::QTensorDialect, scf::SCFDialect,
                  arith::ArithDialect, func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  SmallVector<BenchmarkEntry> entries;
  for (size_t i = 2; i <= 120; ++i) {
    entries.emplace_back("grover", i, [i](MLIRContext* context) {
      return groverDecomposed(context, static_cast<int64_t>(i), 1000,
                              std::string(i, '1'));
    });
    entries.emplace_back("qaoa", i, [i](MLIRContext* context) {
      return qaoa(context, static_cast<int64_t>(i), 100, 0.7, 0.3);
    });
    entries.emplace_back(
        "magic-state-distillation", i, [i](MLIRContext* context) {
          return magicState(context, static_cast<int64_t>(i), 5);
        });
    entries.emplace_back("vqe", i, [i](MLIRContext* context) {
      return vqe(context, static_cast<int64_t>(i), 1000, 0.05);
    });
  }

  SmallVector<AggregateBenchmarkResult> allAggStats;
  for (const auto& entry : entries) {
    llvm::dbgs() << "[benchmark] " << entry.name << " with " << entry.numQubits
                 << " qubits!\n";
    allAggStats.emplace_back(runBenchmark(&context, entry, target, 5));
  }

  printCSV(allAggStats);
  return 0;
}
