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

#include "mlir/Benchmark/Programs.h"
#include "mlir/Compiler/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace mqt::benchmark {

using namespace mlir;

std::optional<QCProgram> generateProgram(const Benchmark& benchmark,
                                         const uint64_t n) {
  // The programs size their registers with signed dimensions, so a size that
  // does not fit into them cannot build a module.
  constexpr auto SIGNED_LIMIT =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const auto upper = benchmark.maximumSize == 0
                         ? SIGNED_LIMIT
                         : std::min(benchmark.maximumSize, SIGNED_LIMIT);

  if (n < benchmark.minimumSize) {
    llvm::errs() << benchmark.name << ": needs a size of at least "
                 << benchmark.minimumSize << "\n";
    return std::nullopt;
  }
  if (n > upper) {
    llvm::errs() << benchmark.name << ": needs a size of at most " << upper
                 << "\n";
    return std::nullopt;
  }

  auto context = createCompilerContext();

  qc::QCProgramBuilder builder(context.get());
  builder.initialize();
  auto results = benchmark.build(builder, n);

  // The initialize call defaults the entry point to an integer result, so the
  // function is retyped to the classical registers the program returns.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(results.size());
  for (auto result : results) {
    resultTypes.emplace_back(result.getType());
  }
  builder.retype(resultTypes);

  auto moduleOp = builder.finalize(results);
  if (!moduleOp) {
    llvm::errs() << benchmark.name << ": failed to build the module\n";
    return std::nullopt;
  }

  auto program = QCProgram::fromModule(context, std::move(moduleOp));
  if (!program || !program->cleanup()) {
    llvm::errs() << benchmark.name << ": failed to clean up the module\n";
    return std::nullopt;
  }
  return program;
}

} // namespace mqt::benchmark
