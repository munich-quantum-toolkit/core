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
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Utils/Graph.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/xxhash.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using Clock = std::chrono::steady_clock;

static CompilerTarget grid(size_t side) {
  std::vector<CompilerTarget::Coupling> edges;
  for (size_t row = 0; row < side; ++row) {
    for (size_t col = 0; col < side; ++col) {
      const auto vertex = row * side + col;
      if (col + 1 < side) {
        edges.emplace_back(vertex, vertex + 1);
      }
      if (row + 1 < side) {
        edges.emplace_back(vertex, vertex + side);
      }
    }
  }
  return llvm::cantFail(CompilerTarget::create(
      side * side, CompilerTarget::Connectivity::fromCouplings(edges),
      CompilerTarget::NativeOperations::unrestricted()));
}

static OwningOpRef<ModuleOp> circuit(MLIRContext& context, bool conditional) {
  QCOProgramBuilder builder(&context);
  builder.initialize();
  SmallVector<Value> qubits;
  for (size_t i = 0; i < (conditional ? 2U : 8U); ++i) {
    qubits.push_back(builder.allocQubit());
  }
  if (conditional) {
    auto [qubit, condition] = builder.measure(qubits[0]);
    qubits[0] = qubit;
    for (size_t i = 0; i < 32; ++i) {
      qubits[1] = builder.qcoIf(condition, qubits[1],
                                [&](Value arg) { return builder.x(arg); });
    }
  } else {
    for (size_t layer = 0; layer < 8; ++layer) {
      for (size_t i = 0; i < qubits.size(); ++i) {
        const auto j = (i + 1 + layer % 3) % qubits.size();
        std::tie(qubits[i], qubits[j]) = builder.cx(qubits[i], qubits[j]);
      }
    }
  }
  for (Value qubit : qubits) {
    builder.sink(qubit);
  }
  return builder.finalize();
}

/// CSV times cover only pass execution; cloning, printing, and verification
/// are outside the timed interval. Every run reports a deterministic IR hash.
int main() {
  MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<mqt::MQTDialect, QCODialect, arith::ArithDialect,
                      func::FuncDialect, scf::SCFDialect>();
  PayloadFormat format;
  format.id = "benchmark.payload";
  format.version = "1.0.0";
  const auto payload =
      llvm::cantFail(PayloadSpecification::create(std::move(format)));
  llvm::outs() << "workload,size,sample,milliseconds,swaps,hash\n";
  for (bool conditional : {true, false}) {
    for (size_t side : {4U, 8U, 16U}) {
      auto input = circuit(context, conditional);
      attachTargetEnvironment(*input, TargetEnvironment(grid(side), payload));
      if (failed(verify(*input))) {
        return 1;
      }
      for (size_t sample = 0; sample < 6; ++sample) {
        OwningOpRef<ModuleOp> moduleOp(input->clone());
        PassManager pm(&context);
        pm.addPass(createMappingPass(
            MappingPassOptions{.niterations = 1, .ntrials = 1, .seed = 42}));
        const auto start = Clock::now();
        const auto result = pm.run(*moduleOp);
        const auto elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        if (failed(result) || failed(verify(*moduleOp))) {
          return 1;
        }
        size_t swaps = 0;
        moduleOp->walk([&](SWAPOp) { ++swaps; });
        std::string ir;
        llvm::raw_string_ostream stream(ir);
        moduleOp->print(stream);
        /// Discard the first run for each workload as a warmup.
        if (sample != 0) {
          llvm::outs() << (conditional ? "conditional" : "routing") << ','
                       << side * side << ',' << sample << ',' << elapsed << ','
                       << swaps << ',' << llvm::xxh3_64bits(ir) << '\n';
        }
      }
    }
  }
  for (size_t size : {128U, 512U, 2048U}) {
    SmallVector<size_t> nodes;
    for (size_t i = 0; i < size; ++i) {
      nodes.push_back(i);
    }
    Graph graph(nodes);
    for (size_t i = 1; i < size; ++i) {
      graph.addEdge(0, i);
    }
    for (size_t sample = 0; sample < 6; ++sample) {
      const auto start = Clock::now();
      for (size_t iteration = 0; iteration < 100; ++iteration) {
        if (graph.findCycle()) {
          return 1;
        }
      }
      const auto elapsed =
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      if (sample != 0) {
        llvm::outs() << "graph," << size << ',' << sample << ',' << elapsed
                     << ",0,0\n";
      }
    }
  }
}
