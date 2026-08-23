/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Generate.h"

#include "benchmarks/GHZ.hpp"
#include "benchmarks/Grover.hpp"
#include "benchmarks/QPE.hpp"
#include "mlir/Compiler/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "programs/Programs.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

#include <optional>
#include <utility>

namespace mqt::benchmark {

using namespace mlir;

namespace {

[[nodiscard]] std::optional<QCProgram> buildProgram(
    const llvm::StringRef name,
    const llvm::function_ref<SmallVector<Value>(qc::QCProgramBuilder&)>& emit) {
  auto context = createCompilerContext();
  auto moduleOp = qc::QCProgramBuilder::build(context.get(), emit);
  if (!moduleOp) {
    llvm::errs() << name << ": failed to build the module\n";
    return std::nullopt;
  }

  auto program = QCProgram::fromModule(context, std::move(moduleOp));
  if (!program || !program->cleanup()) {
    llvm::errs() << name << ": failed to clean up the module\n";
    return std::nullopt;
  }
  return program;
}

} // namespace

std::optional<QCProgram> generateProgram(const benchmarks::GHZ& benchmark) {
  return buildProgram(
      "ghz", [&](qc::QCProgramBuilder& b) { return ghz(b, benchmark); });
}

std::optional<QCProgram> generateProgram(const benchmarks::Grover& benchmark) {
  return buildProgram(
      "grover", [&](qc::QCProgramBuilder& b) { return grover(b, benchmark); });
}

std::optional<QCProgram> generateProgram(const benchmarks::QPE& benchmark) {
  return buildProgram(
      "qpe", [&](qc::QCProgramBuilder& b) { return qpe(b, benchmark); });
}

} // namespace mqt::benchmark
