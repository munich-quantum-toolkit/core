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

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static FailureOr<QCProgram> buildProgram(
    const llvm::StringRef name,
    const llvm::function_ref<SmallVector<Value>(qc::QCProgramBuilder&)>& emit) {
  auto context = createCompilerContext();
  std::string diagnostics;
  const mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic& diagnostic) {
        if (!diagnostics.empty()) {
          diagnostics.push_back('\n');
        }
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return success();
      });
  auto moduleOp = qc::QCProgramBuilder::build(context.get(), emit);
  if (!moduleOp) {
    return ::mqt::emitError(
        llvm::Twine(name + ": failed to build the module: " + diagnostics)
            .str(),
        ::mqt::ErrorCategory::InvalidArgument);
  }

  auto program = QCProgram::fromModule(context, std::move(moduleOp));
  if (!program || !program->cleanup()) {
    return ::mqt::emitError(
        llvm::Twine(name + ": failed to clean up the module: " + diagnostics)
            .str(),
        ::mqt::ErrorCategory::InvalidArgument);
  }
  return std::move(*program);
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  FailureOr<QCProgram> generate(const TYPE& benchmark) {                       \
    return buildProgram(ID, [&](qc::QCProgramBuilder& builder) {               \
      return STEM(builder, benchmark);                                         \
    });                                                                        \
  }
#include "bench/BenchmarkFamilies.inc"

FailureOr<GeneratedBenchmark>
generate(const std::string_view instanceSpecificationJSON,
         const std::string_view source) {
  auto result =
      parseInstanceSpecificationJSON(instanceSpecificationJSON, source);
  if (failed(result)) {
    return failure();
  }
  auto& parsed = *result;
  auto program =
      std::visit([](const auto& benchmark) { return generate(benchmark); },
                 parsed.instance);
  if (failed(program)) {
    return failure();
  }
  return GeneratedBenchmark{
      .benchmarkId = std::move(parsed.benchmarkId),
      .caseId = std::move(parsed.caseId),
      .manifestJSON = std::move(parsed.manifestJSON),
      .program = std::move(*program),
  };
}

} // namespace mqt::bench
