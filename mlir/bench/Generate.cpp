/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/bench/Generate.h"

#include "bench/JSON.hpp"
#include "mlir/Compiler/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "programs/Programs.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

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

template <class Benchmark>
[[nodiscard]] static std::optional<GeneratedBenchmark>
generateInstance(const std::string_view id, const Benchmark& benchmark) {
  auto program = generate(benchmark);
  if (!program) {
    return std::nullopt;
  }
  return GeneratedBenchmark{
      std::string(id),
      caseId(benchmark),
      toManifestJSON(benchmark),
      std::move(*program),
  };
}

using GenerateFunction =
    std::optional<GeneratedBenchmark> (*)(std::string_view, std::string_view);

namespace {
struct RegistryEntry {
  std::string_view id;
  GenerateFunction generate;
};
} // namespace

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  RegistryEntry{ID, [](const std::string_view instanceSpecificationJSON,       \
                       const std::string_view source) {                        \
                  return generateInstance(                                     \
                      ID, STEM##FromInstanceSpecificationJSON(                 \
                              instanceSpecificationJSON, source));             \
                }},
static const std::array REGISTRY{
#include "bench/BenchmarkFamilies.inc"
};

std::optional<GeneratedBenchmark>
generate(const std::string_view instanceSpecificationJSON,
         const std::string_view source) {
  const auto id = benchmarkIdFromInstanceSpecificationJSON(
      instanceSpecificationJSON, source);
  for (const auto& entry : REGISTRY) {
    if (entry.id == id) {
      return entry.generate(instanceSpecificationJSON, source);
    }
  }
  return std::nullopt;
}

} // namespace mqt::bench
