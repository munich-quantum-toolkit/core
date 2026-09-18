/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/bench/Generate.h"

#include "bench/JSON.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "programs/Programs.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static std::optional<QCProgram> buildProgram(
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

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  std::optional<QCProgram> generate(const TYPE& benchmark) {                   \
    return buildProgram(ID, [&](qc::QCProgramBuilder& builder) {               \
      return STEM(builder, benchmark);                                         \
    });                                                                        \
  }
#include "bench/BenchmarkFamilies.inc"

std::optional<GeneratedBenchmark>
generate(const std::string_view instanceSpecificationJSON,
         const std::string_view source) {
  auto result =
      tryParseInstanceSpecificationJSON(instanceSpecificationJSON, source);
  if (const auto* error = std::get_if<JSONError>(&result)) {
    llvm::errs() << error->message << '\n';
    return std::nullopt;
  }
  auto& parsed = std::get<ParsedBenchmark>(result);
  auto program =
      std::visit([](const auto& benchmark) { return generate(benchmark); },
                 parsed.instance);
  if (!program) {
    return std::nullopt;
  }
  return GeneratedBenchmark{
      .benchmarkId = std::move(parsed.benchmarkId),
      .caseId = std::move(parsed.caseId),
      .manifestJSON = std::move(parsed.manifestJSON),
      .program = std::move(*program),
  };
}

} // namespace mqt::bench
