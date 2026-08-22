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

#include "benchmarks/Evaluation.hpp"
#include "benchmarks/GHZ.hpp"
#include "benchmarks/Grover.hpp"
#include "benchmarks/QPE.hpp"
#include "benchmarks/mqt_core_benchmarks_export.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace mqt::benchmarks {

/// Return the benchmark ID from a strict version-one request.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
benchmarkIdFromRequestJSON(std::string_view json,
                           std::string_view source = "<request>");

/// Return the benchmark ID from a strict version-one manifest envelope.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
benchmarkIdFromManifestJSON(std::string_view json,
                            std::string_view source = "<manifest>");

/// Return the fixed benchmark registry as canonical JSON.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string listBenchmarksJSON();

/// Return the version-one request schema for a benchmark as canonical JSON.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
describeBenchmarkJSON(std::string_view benchmark);

[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT GHZ ghzFromRequestJSON(
    std::string_view json, std::string_view source = "<request>");
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT Grover groverFromRequestJSON(
    std::string_view json, std::string_view source = "<request>");
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT QPE qpeFromRequestJSON(
    std::string_view json, std::string_view source = "<request>");

[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
toRequestJSON(const GHZ& benchmark);
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
toRequestJSON(const Grover& benchmark);
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
toRequestJSON(const QPE& benchmark);

[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT GHZ ghzFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT Grover groverFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT QPE qpeFromManifestJSON(
    std::string_view json, std::string_view source = "<manifest>");

[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
toManifestJSON(const GHZ& benchmark);
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
toManifestJSON(const Grover& benchmark);
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
toManifestJSON(const QPE& benchmark);

/// Return the stable semantic case ID of a benchmark instance.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
caseId(const GHZ& benchmark);
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
caseId(const Grover& benchmark);
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
caseId(const QPE& benchmark);

/// Parse sampled outcomes from a strict version-one counts document.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT Counts
countsFromJSON(std::string_view json, std::string_view source = "<counts>");

/// Serialize one evaluation result as canonical JSON.
[[nodiscard]] MQT_CORE_BENCHMARKS_EXPORT std::string
evaluationToJSON(std::string_view caseId, size_t shots,
                 const Evaluation& evaluation);

} // namespace mqt::benchmarks
