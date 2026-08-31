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
#include "bench/QFT.hpp"
#include "bench/QPE.hpp"
#include "bench/mqt_core_bench_export.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace mqt::bench {

/// Return the benchmark ID from a strict instance specification.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
benchmarkIdFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");

/// Return the benchmark ID from a strict manifest envelope.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
benchmarkIdFromManifestJSON(std::string_view json,
                            std::string_view source = "<manifest>");

/// Return the fixed benchmark registry as canonical JSON.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string listBenchmarksJSON();

/// Return the instance specification schema for a benchmark as canonical JSON.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
describeBenchmarkJSON(std::string_view benchmark);

[[nodiscard]] MQT_CORE_BENCH_EXPORT BV bvFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT GHZ ghzFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT Grover groverFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT QFT qftFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT QPE qpeFromInstanceSpecificationJSON(
    std::string_view json,
    std::string_view source = "<instance-specification>");

[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toInstanceSpecificationJSON(const BV& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toInstanceSpecificationJSON(const GHZ& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toInstanceSpecificationJSON(const Grover& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toInstanceSpecificationJSON(const QFT& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toInstanceSpecificationJSON(const QPE& benchmark);

[[nodiscard]] MQT_CORE_BENCH_EXPORT BV bvFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT GHZ ghzFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT Grover groverFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT QFT qftFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");
[[nodiscard]] MQT_CORE_BENCH_EXPORT QPE qpeFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");

[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toManifestJSON(const BV& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toManifestJSON(const GHZ& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toManifestJSON(const Grover& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toManifestJSON(const QFT& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
toManifestJSON(const QPE& benchmark);

/// Return the stable semantic case ID of a benchmark instance.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string caseId(const BV& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string caseId(const GHZ& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string caseId(const Grover& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string caseId(const QFT& benchmark);
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string caseId(const QPE& benchmark);

/// Parse sampled outcomes from a strict counts document.
[[nodiscard]] MQT_CORE_BENCH_EXPORT Counts
countsFromJSON(std::string_view json, std::string_view source = "<counts>");

/// Evaluate a counts document against a benchmark manifest.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
evaluateJSON(std::string_view manifest, std::string_view counts,
             std::string_view manifestSource = "<manifest>",
             std::string_view countsSource = "<counts>");

/// Serialize one evaluation result as canonical JSON.
[[nodiscard]] MQT_CORE_BENCH_EXPORT std::string
evaluationToJSON(std::string_view caseId, size_t shots,
                 const Evaluation& evaluation);

} // namespace mqt::bench
