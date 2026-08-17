/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {

llvm::cl::opt<std::string>
    scenario("scenario", llvm::cl::desc("Workload: frontier or routing"),
             llvm::cl::value_desc("name"), llvm::cl::init("frontier"));
llvm::cl::opt<size_t> nqubits("qubits", llvm::cl::desc("Number of qubits"),
                              llvm::cl::init(36));
llvm::cl::opt<size_t> nlayers("layers", llvm::cl::desc("Number of gate layers"),
                              llvm::cl::init(120));
llvm::cl::opt<size_t> seed("seed", llvm::cl::desc("Deterministic random seed"),
                           llvm::cl::init(1930));
llvm::cl::opt<size_t>
    nlookahead("lookahead", llvm::cl::desc("Number of mapping lookahead steps"),
               llvm::cl::init(20));
llvm::cl::opt<float> lambda("lambda",
                            llvm::cl::desc("Mapping cost decay factor"),
                            llvm::cl::init(0.5F));
llvm::cl::opt<size_t>
    niterations("iterations",
                llvm::cl::desc("Number of mapping refinement iterations"),
                llvm::cl::init(1));
llvm::cl::opt<size_t> ntrials("trials",
                              llvm::cl::desc("Number of initial layout trials"),
                              llvm::cl::init(18));

using CouplingSet = llvm::DenseSet<std::pair<size_t, size_t>>;

CouplingSet makeGrid(const size_t count) {
  const auto width =
      static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(count))));
  CouplingSet couplingSet;
  for (size_t i = 0; i < count; ++i) {
    if (i % width + 1 < width && i + 1 < count) {
      couplingSet.insert({i, i + 1});
      couplingSet.insert({i + 1, i});
    }
    if (i + width < count) {
      couplingSet.insert({i, i + width});
      couplingSet.insert({i + width, i});
    }
  }
  return couplingSet;
}

OwningOpRef<ModuleOp> makeProgram(MLIRContext& context) {
  QCOProgramBuilder builder(&context);
  builder.initialize(llvm::SmallVector<Type>(nqubits, builder.getI1Type()));

  Value tensor = builder.qtensorAlloc(static_cast<int64_t>(nqubits));
  llvm::SmallVector<Value> qubits(nqubits);
  llvm::SmallVector<Value> bits(nqubits);
  for (size_t i = 0; i < nqubits; ++i) {
    std::tie(tensor, qubits[i]) =
        builder.qtensorExtract(tensor, static_cast<int64_t>(i));
  }

  std::mt19937_64 rng(seed);
  llvm::SmallVector<size_t> order(nqubits);
  std::iota(order.begin(), order.end(), 0);

  if (scenario == "frontier") {
    for (size_t layer = 0; layer < nlayers; ++layer) {
      std::ranges::shuffle(order, rng);
      for (size_t i = 0; i + 1 < order.size(); i += 2) {
        const auto first = order[i];
        const auto second = order[i + 1];
        if (layer % 2 == 0) {
          std::tie(qubits[first], qubits[second]) =
              builder.cx(qubits[first], qubits[second]);
        } else {
          std::tie(qubits[first], qubits[second]) =
              builder.cz(qubits[first], qubits[second]);
        }
      }
    }
  } else if (scenario == "routing") {
    size_t active = 0;
    for (size_t layer = 0; layer < nlayers; ++layer) {
      size_t target = active;
      while (target == active) {
        target = rng() % nqubits;
      }
      std::tie(qubits[active], qubits[target]) =
          builder.cx(qubits[active], qubits[target]);
      active = target;
      qubits[active] = builder.h(qubits[active]);
    }
  } else {
    llvm::errs() << "unknown scenario: " << scenario << '\n';
    return {};
  }

  qubits = builder.barrier(qubits);
  for (size_t i = 0; i < nqubits; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
    tensor = builder.qtensorInsert(qubits[i], tensor, static_cast<int64_t>(i));
  }
  builder.qtensorDealloc(tensor);
  return builder.finalize(bits);
}

} // namespace

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (nqubits < 2) {
    llvm::errs() << "--qubits must be at least 2\n";
    return 2;
  }

  DialectRegistry registry;
  registry.insert<QCODialect, scf::SCFDialect, arith::ArithDialect,
                  func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = makeProgram(context);
  if (!module) {
    return 2;
  }

  PassManager pm(&context);
  pm.enableVerifier(false);
  pm.addPass(
      createMappingPass(makeGrid(nqubits), MappingPassOptions{
                                               .nlookahead = nlookahead,
                                               .lambda = lambda,
                                               .niterations = niterations,
                                               .ntrials = ntrials,
                                               .seed = seed,
                                           }));

  const auto start = std::chrono::steady_clock::now();
  const auto result = pm.run(module.get());
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start);
  if (failed(result)) {
    llvm::errs() << "mapping failed\n";
    return 1;
  }
  if (failed(verify(module.get()))) {
    llvm::errs() << "mapped module verification failed\n";
    return 1;
  }
  std::cout << elapsed.count() << '\n';
  return 0;
}
