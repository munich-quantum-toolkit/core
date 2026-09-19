/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "bench/BV.hpp"
#include "bench/Evaluation.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Teleportation.hpp"
#include "bench/mqt_core_bench_export.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <variant>

namespace mqt::bench {

/// One validated benchmark instance from the JSON registry.
using BenchmarkInstance =
    std::variant<BV, GHZ, Grover, ModularMultiplier, Multiplexer, QFT, QFTAdder,
                 QPE, RepeatUntilSuccess, Teleportation>;

/// A validated instance and its normalized metadata, ready for generation.
struct ParsedBenchmark {
  BenchmarkInstance instance;
  std::string benchmarkId;
  std::string caseId;
  std::string manifestJSON;
};

/// Parse and normalize a benchmark, returning a diagnostic on invalid input.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<ParsedBenchmark>
parseInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");

/// Return the benchmark ID from a strict instance specification.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string>
benchmarkIdFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");

/// Return the benchmark ID from a strict manifest envelope.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string>
benchmarkIdFromManifestJSON(std::string_view json,
                            std::string_view source = "<manifest>");

/// Return the fixed benchmark registry as canonical JSON.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string listBenchmarksJSON();

/// Return the instance specification schema for a benchmark as canonical JSON.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string>
describeBenchmarkJSON(std::string_view benchmark);

/// Family-specific JSON conversions and stable semantic case IDs.
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  [[nodiscard]] MQT_CORE_BENCH_EXPORT Result<TYPE>                             \
  STEM##FromInstanceSpecificationJSON(std::string_view json,                   \
                                      std::string_view source =                \
                                          "<instance-specification>");         \
  [[nodiscard]] MQT_CORE_BENCH_EXPORT std::string toInstanceSpecificationJSON( \
      const TYPE& benchmark);                                                  \
  [[nodiscard]] MQT_CORE_BENCH_EXPORT Result<TYPE> STEM##FromManifestJSON(     \
      std::string_view json, std::string_view source = "<manifest>");          \
  [[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string> toManifestJSON(      \
      const TYPE& benchmark);                                                  \
  [[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string> caseId(              \
      const TYPE& benchmark);
#include "bench/BenchmarkFamilies.inc"

/// Parse sampled outcomes from a strict counts document.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<Counts>
countsFromJSON(std::string_view json, std::string_view source = "<counts>");

/// Evaluate a counts document against a benchmark manifest.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string>
evaluateJSON(std::string_view manifest, std::string_view counts,
             std::string_view manifestSource = "<manifest>",
             std::string_view countsSource = "<counts>");

/// Serialize one evaluation result as canonical JSON.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Result<std::string>
evaluationToJSON(std::string_view caseId, size_t shots,
                 const Evaluation& evaluation);

} // namespace mqt::bench
