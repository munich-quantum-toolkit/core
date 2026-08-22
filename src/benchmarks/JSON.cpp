/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/JSON.hpp"

#include "SHA256.hpp"

#include <nlohmann/json.hpp> // NOLINT(misc-include-cleaner)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mqt::benchmarks {
namespace {

using Json = nlohmann::json;

constexpr uint64_t SCHEMA_VERSION = 1;
constexpr uint64_t DEFINITION_VERSION = 1;
constexpr std::string_view CASE_DOMAIN = "mqt-core:benchmark-case:v1";
constexpr std::array<std::string_view, 3> BENCHMARK_IDS{"ghz", "grover", "qpe"};

[[noreturn]] void fail(const std::string_view source,
                       const std::string_view pointer,
                       const std::string_view message) {
  throw std::invalid_argument(std::string(source) + ":" + std::string(pointer) +
                              " " + std::string(message));
}

[[nodiscard]] Json parseJSON(const std::string_view text,
                             const std::string_view source) {
  std::vector<std::vector<std::string>> keysByDepth;
  const Json::parser_callback_t rejectDuplicates =
      [&](const int depth, const Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
          const auto index = static_cast<size_t>(depth);
          if (keysByDepth.size() <= index) {
            keysByDepth.resize(index + 1U);
          }
          keysByDepth[index].clear();
        } else if (event == Json::parse_event_t::key) {
          const auto index = static_cast<size_t>(depth - 1);
          const auto& key = parsed.get_ref<const std::string&>();
          auto& keys = keysByDepth.at(index);
          if (std::ranges::find(keys, key) != keys.end()) {
            fail(source, "$", "contains duplicate key '" + key + "'");
          }
          keys.emplace_back(key);
        }
        return true;
      };

  try {
    return Json::parse(text.begin(), text.end(), rejectDuplicates);
  } catch (const Json::exception& error) {
    throw std::invalid_argument(std::string(source) +
                                ": invalid JSON: " + error.what());
  }
}

void requireObject(const Json& value, const std::string_view source,
                   const std::string_view pointer) {
  if (!value.is_object()) {
    fail(source, pointer, "must be an object");
  }
}

void rejectUnknownKeys(const Json& value,
                       const std::initializer_list<std::string_view> known,
                       const std::string_view source,
                       const std::string_view pointer) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(known, key) == known.end()) {
      fail(source, pointer, "contains unknown key '" + key + "'");
    }
  }
}

[[nodiscard]] const Json& required(const Json& value, const char* const key,
                                   const std::string_view source,
                                   const std::string_view pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    fail(source, std::string(pointer) + "/" + key, "is required");
  }
  return *found;
}

[[nodiscard]] uint64_t unsignedInteger(const Json& value,
                                       const std::string_view source,
                                       const std::string_view pointer) {
  if (value.is_number_float()) {
    const auto parsed = value.get<double>();
    if (!std::isfinite(parsed) || parsed < 0. || std::trunc(parsed) != parsed ||
        parsed >= std::ldexp(1., 64)) {
      fail(source, pointer, "must be a non-negative integer");
    }
    return static_cast<uint64_t>(parsed);
  }
  if (!value.is_number_unsigned() &&
      (!value.is_number_integer() || value.get<int64_t>() < 0)) {
    fail(source, pointer, "must be a non-negative integer");
  }
  try {
    return value.get<uint64_t>();
  } catch (const Json::exception&) {
    fail(source, pointer, "must fit an unsigned 64-bit integer");
  }
}

[[nodiscard]] size_t sizeValue(const Json& value, const std::string_view source,
                               const std::string_view pointer) {
  const auto parsed = unsignedInteger(value, source, pointer);
  if (parsed > std::numeric_limits<size_t>::max()) {
    fail(source, pointer, "must fit size_t");
  }
  return static_cast<size_t>(parsed);
}

[[nodiscard]] std::string stringValue(const Json& value,
                                      const std::string_view source,
                                      const std::string_view pointer) {
  if (!value.is_string()) {
    fail(source, pointer, "must be a string");
  }
  return value.get<std::string>();
}

void requireSchemaVersion(const Json& root, const std::string_view source) {
  const auto version =
      unsignedInteger(required(root, "schema_version", source, "$"), source,
                      "$/schema_version");
  if (version != SCHEMA_VERSION) {
    fail(source, "$/schema_version", "must be 1");
  }
}

[[nodiscard]] bool isKnownBenchmark(const std::string_view benchmark) {
  return std::ranges::find(BENCHMARK_IDS, benchmark) != BENCHMARK_IDS.end();
}

[[nodiscard]] std::string requireBenchmarkId(const Json& root,
                                             const std::string_view source) {
  auto benchmark = stringValue(required(root, "benchmark", source, "$"), source,
                               "$/benchmark");
  if (!isKnownBenchmark(benchmark)) {
    fail(source, "$/benchmark",
         "selects unsupported benchmark '" + benchmark + "'");
  }
  return benchmark;
}

[[nodiscard]] Json requestEnvelope(const std::string_view text,
                                   const std::string_view source) {
  auto root = parseJSON(text, source);
  requireObject(root, source, "$");
  rejectUnknownKeys(root, {"schema_version", "benchmark", "parameters"}, source,
                    "$");
  requireSchemaVersion(root, source);
  static_cast<void>(requireBenchmarkId(root, source));
  requireObject(required(root, "parameters", source, "$"), source,
                "$/parameters");
  return root;
}

[[nodiscard]] Json manifestEnvelope(const std::string_view text,
                                    const std::string_view source) {
  auto root = parseJSON(text, source);
  requireObject(root, source, "$");
  rejectUnknownKeys(root,
                    {"schema_version", "case_id", "benchmark",
                     "definition_version", "parameters", "outputs",
                     "reference"},
                    source, "$");
  requireSchemaVersion(root, source);
  static_cast<void>(requireBenchmarkId(root, source));
  const auto definition =
      unsignedInteger(required(root, "definition_version", source, "$"), source,
                      "$/definition_version");
  if (definition != DEFINITION_VERSION) {
    fail(source, "$/definition_version", "must be 1");
  }
  static_cast<void>(
      stringValue(required(root, "case_id", source, "$"), source, "$/case_id"));
  requireObject(required(root, "parameters", source, "$"), source,
                "$/parameters");
  if (!required(root, "outputs", source, "$").is_array()) {
    fail(source, "$/outputs", "must be an array");
  }
  requireObject(required(root, "reference", source, "$"), source,
                "$/reference");
  return root;
}

void requireBenchmark(const Json& root, const std::string_view expected,
                      const std::string_view source) {
  const auto actual = stringValue(required(root, "benchmark", source, "$"),
                                  source, "$/benchmark");
  if (actual != expected) {
    fail(source, "$/benchmark", "must be '" + std::string(expected) + "'");
  }
}

[[nodiscard]] GHZ parseGHZParameters(const Json& parameters,
                                     const std::string_view source) {
  rejectUnknownKeys(parameters, {"qubits", "topology", "basis"}, source,
                    "$/parameters");
  GHZOptions options{.qubits = sizeValue(
                         required(parameters, "qubits", source, "$/parameters"),
                         source, "$/parameters/qubits")};
  if (const auto topology = parameters.find("topology");
      topology != parameters.end()) {
    const auto value = stringValue(*topology, source, "$/parameters/topology");
    if (value == "linear") {
      options.topology = GHZTopology::Linear;
    } else if (value == "star") {
      options.topology = GHZTopology::Star;
    } else {
      fail(source, "$/parameters/topology", "must be 'linear' or 'star'");
    }
  }
  if (const auto basis = parameters.find("basis"); basis != parameters.end()) {
    const auto value = stringValue(*basis, source, "$/parameters/basis");
    if (value == "z") {
      options.basis = GHZBasis::Z;
    } else if (value == "x") {
      options.basis = GHZBasis::X;
    } else {
      fail(source, "$/parameters/basis", "must be 'z' or 'x'");
    }
  }
  try {
    return GHZ(options);
  } catch (const std::invalid_argument& error) {
    fail(source, "$/parameters", error.what());
  }
}

[[nodiscard]] Grover parseGroverParameters(const Json& parameters,
                                           const std::string_view source) {
  rejectUnknownKeys(parameters, {"marked_bitstring", "iterations"}, source,
                    "$/parameters");
  GroverOptions options{
      .markedBitstring = stringValue(
          required(parameters, "marked_bitstring", source, "$/parameters"),
          source, "$/parameters/marked_bitstring")};
  if (const auto iterations = parameters.find("iterations");
      iterations != parameters.end()) {
    options.iterations =
        sizeValue(*iterations, source, "$/parameters/iterations");
  }
  try {
    return Grover(std::move(options));
  } catch (const std::invalid_argument& error) {
    fail(source, "$/parameters", error.what());
  }
}

[[nodiscard]] QPE parseQPEParameters(const Json& parameters,
                                     const std::string_view source) {
  rejectUnknownKeys(parameters, {"precision", "phase", "method"}, source,
                    "$/parameters");
  const auto precision =
      sizeValue(required(parameters, "precision", source, "$/parameters"),
                source, "$/parameters/precision");
  const auto& phase = required(parameters, "phase", source, "$/parameters");
  requireObject(phase, source, "$/parameters/phase");
  rejectUnknownKeys(phase, {"numerator", "denominator"}, source,
                    "$/parameters/phase");
  const auto numerator = unsignedInteger(
      required(phase, "numerator", source, "$/parameters/phase"), source,
      "$/parameters/phase/numerator");
  const auto denominator = unsignedInteger(
      required(phase, "denominator", source, "$/parameters/phase"), source,
      "$/parameters/phase/denominator");
  auto method = QPEMethod::Standard;
  if (const auto value = parameters.find("method"); value != parameters.end()) {
    const auto name = stringValue(*value, source, "$/parameters/method");
    if (name == "standard") {
      method = QPEMethod::Standard;
    } else if (name == "iterative") {
      method = QPEMethod::Iterative;
    } else {
      fail(source, "$/parameters/method", "must be 'standard' or 'iterative'");
    }
  }
  try {
    return QPE({.precision = precision,
                .phase = Phase(numerator, denominator),
                .method = method});
  } catch (const std::invalid_argument& error) {
    fail(source, "$/parameters", error.what());
  }
}

[[nodiscard]] std::string topologyName(const GHZTopology topology) {
  return topology == GHZTopology::Linear ? "linear" : "star";
}

[[nodiscard]] std::string basisName(const GHZBasis basis) {
  return basis == GHZBasis::Z ? "z" : "x";
}

[[nodiscard]] std::string methodName(const QPEMethod method) {
  return method == QPEMethod::Standard ? "standard" : "iterative";
}

[[nodiscard]] Json parametersJSON(const GHZ& benchmark) {
  const auto& options = benchmark.options();
  return {{"basis", basisName(options.basis)},
          {"qubits", options.qubits},
          {"topology", topologyName(options.topology)}};
}

[[nodiscard]] Json parametersJSON(const Grover& benchmark) {
  const auto& options = benchmark.options();
  return {{"iterations", *options.iterations},
          {"marked_bitstring", options.markedBitstring}};
}

[[nodiscard]] Json parametersJSON(const QPE& benchmark) {
  const auto& options = benchmark.options();
  return {{"method", methodName(options.method)},
          {"phase",
           {{"denominator", options.phase.denominator()},
            {"numerator", options.phase.numerator()}}},
          {"precision", options.precision}};
}

[[nodiscard]] Json referenceJSON(const GHZ& benchmark) {
  return {{"kind", "analytic"},
          {"model", "ghz"},
          {"outcome_order", "big_endian"},
          {"output", benchmark.output().name},
          {"version", 1}};
}

[[nodiscard]] Json referenceJSON(const Grover& benchmark) {
  return {{"kind", "analytic"},
          {"model", "grover_single_marked"},
          {"outcome_order", "big_endian"},
          {"output", benchmark.output().name},
          {"success_outcome", benchmark.options().markedBitstring},
          {"version", 1}};
}

[[nodiscard]] Json referenceJSON(const QPE& benchmark) {
  return {{"kind", "analytic"},
          {"model", "qpe_dirichlet"},
          {"outcome_order", "big_endian"},
          {"output", benchmark.output().name},
          {"version", 1}};
}

[[nodiscard]] Json semanticJSON(const std::string_view id,
                                const Json& parameters, const Output& output,
                                const Json& reference) {
  return {{"benchmark", std::string(id)},
          {"definition_version", DEFINITION_VERSION},
          {"outputs",
           Json::array({{{"name", output.name}, {"width", output.width}}})},
          {"parameters", parameters},
          {"reference", reference}};
}

[[nodiscard]] Json semanticJSON(const GHZ& benchmark) {
  return semanticJSON("ghz", parametersJSON(benchmark), benchmark.output(),
                      referenceJSON(benchmark));
}

[[nodiscard]] Json semanticJSON(const Grover& benchmark) {
  return semanticJSON("grover", parametersJSON(benchmark), benchmark.output(),
                      referenceJSON(benchmark));
}

[[nodiscard]] Json semanticJSON(const QPE& benchmark) {
  return semanticJSON("qpe", parametersJSON(benchmark), benchmark.output(),
                      referenceJSON(benchmark));
}

[[nodiscard]] std::string semanticCaseId(const Json& semantic) {
  auto input = std::string(CASE_DOMAIN);
  input.push_back('\0');
  input += semantic.dump();
  return "sha256-" + detail::sha256Hex(input);
}

template <class Benchmark>
[[nodiscard]] Json manifestJSON(const Benchmark& benchmark) {
  auto semantic = semanticJSON(benchmark);
  semantic["case_id"] = semanticCaseId(semantic);
  semantic["schema_version"] = SCHEMA_VERSION;
  return semantic;
}

[[nodiscard]] Json requestJSON(const std::string_view id,
                               const Json& parameters) {
  return {{"benchmark", std::string(id)},
          {"parameters", parameters},
          {"schema_version", SCHEMA_VERSION}};
}

template <class Benchmark, class ParseParameters>
[[nodiscard]] Benchmark parseManifest(const std::string_view text,
                                      const std::string_view source,
                                      const std::string_view expectedId,
                                      ParseParameters&& parseParameters) {
  const auto root = manifestEnvelope(text, source);
  requireBenchmark(root, expectedId, source);
  auto benchmark = parseParameters(root.at("parameters"), source);
  if (root.dump() != manifestJSON(benchmark).dump()) {
    fail(source, "$",
         "does not match its resolved benchmark instance and case ID");
  }
  return benchmark;
}

[[nodiscard]] Json baseRequestSchema(const std::string_view id,
                                     Json parameters) {
  return {{"$schema", "https://json-schema.org/draft/2020-12/schema"},
          {"additionalProperties", false},
          {"properties",
           {{"benchmark", {{"const", std::string(id)}}},
            {"parameters", std::move(parameters)},
            {"schema_version", {{"const", SCHEMA_VERSION}}}}},
          {"required", {"schema_version", "benchmark", "parameters"}},
          {"type", "object"},
          {"x-mqt-definition-version", DEFINITION_VERSION}};
}

[[nodiscard]] Json ghzSchema() {
  Json parameters{
      {"additionalProperties", false},
      {"properties",
       {{"basis", {{"default", "z"}, {"enum", {"z", "x"}}}},
        {"qubits",
         {{"maximum", GHZOptions::MAX_QUBITS},
          {"minimum", 1},
          {"type", "integer"}}},
        {"topology", {{"default", "linear"}, {"enum", {"linear", "star"}}}}}},
      {"required", {"qubits"}},
      {"type", "object"}};
  parameters["allOf"] = Json::array(
      {{{"if",
         {{"properties", {{"basis", {{"const", "x"}}}}},
          {"required", {"basis"}}}},
        {"then",
         {{"properties",
           {{"qubits", {{"maximum", GHZOptions::MAX_X_BASIS_QUBITS}}}}}}}}});
  return baseRequestSchema("ghz", std::move(parameters));
}

[[nodiscard]] Json groverSchema() {
  return baseRequestSchema(
      "grover", {{"additionalProperties", false},
                 {"properties",
                  {{"iterations",
                    {{"maximum", std::numeric_limits<int32_t>::max()},
                     {"minimum", 0},
                     {"type", "integer"}}},
                   {"marked_bitstring",
                    {{"maxLength", 62},
                     {"minLength", 2},
                     {"pattern", "^[01]+$"},
                     {"type", "string"}}}}},
                 {"required", {"marked_bitstring"}},
                 {"type", "object"}});
}

[[nodiscard]] Json qpeSchema() {
  return baseRequestSchema(
      "qpe",
      {{"additionalProperties", false},
       {"properties",
        {{"method",
          {{"default", "standard"}, {"enum", {"standard", "iterative"}}}},
         {"phase",
          {{"additionalProperties", false},
           {"properties",
            {{"denominator",
              {{"maximum", std::numeric_limits<uint64_t>::max()},
               {"minimum", 1},
               {"type", "integer"}}},
             {"numerator",
              {{"maximum", std::numeric_limits<uint64_t>::max()},
               {"minimum", 0},
               {"type", "integer"}}}}},
           {"required", {"numerator", "denominator"}},
           {"type", "object"}}},
         {"precision",
          {{"maximum", QPEOptions::MAX_PRECISION},
           {"minimum", 1},
           {"type", "integer"}}}}},
       {"required", {"precision", "phase"}},
       {"type", "object"}});
}

[[nodiscard]] bool validCaseId(const std::string_view value) {
  constexpr std::string_view prefix = "sha256-";
  if (!value.starts_with(prefix) || value.size() != prefix.size() + 64U) {
    return false;
  }
  return std::ranges::all_of(value.substr(prefix.size()), [](const char digit) {
    return (digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f');
  });
}

} // namespace

std::string benchmarkIdFromRequestJSON(const std::string_view json,
                                       const std::string_view source) {
  return requireBenchmarkId(requestEnvelope(json, source), source);
}

std::string benchmarkIdFromManifestJSON(const std::string_view json,
                                        const std::string_view source) {
  return requireBenchmarkId(manifestEnvelope(json, source), source);
}

std::string listBenchmarksJSON() {
  auto benchmarks = Json::array();
  for (const auto id : BENCHMARK_IDS) {
    benchmarks.emplace_back(Json{{"definition_version", DEFINITION_VERSION},
                                 {"id", std::string(id)}});
  }
  return Json{{"benchmarks", std::move(benchmarks)},
              {"schema_version", SCHEMA_VERSION}}
      .dump();
}

std::string describeBenchmarkJSON(const std::string_view benchmark) {
  if (benchmark == "ghz") {
    return ghzSchema().dump();
  }
  if (benchmark == "grover") {
    return groverSchema().dump();
  }
  if (benchmark == "qpe") {
    return qpeSchema().dump();
  }
  throw std::invalid_argument("unsupported benchmark '" +
                              std::string(benchmark) + "'");
}

GHZ ghzFromRequestJSON(const std::string_view json,
                       const std::string_view source) {
  const auto root = requestEnvelope(json, source);
  requireBenchmark(root, "ghz", source);
  return parseGHZParameters(root.at("parameters"), source);
}

Grover groverFromRequestJSON(const std::string_view json,
                             const std::string_view source) {
  const auto root = requestEnvelope(json, source);
  requireBenchmark(root, "grover", source);
  return parseGroverParameters(root.at("parameters"), source);
}

QPE qpeFromRequestJSON(const std::string_view json,
                       const std::string_view source) {
  const auto root = requestEnvelope(json, source);
  requireBenchmark(root, "qpe", source);
  return parseQPEParameters(root.at("parameters"), source);
}

std::string toRequestJSON(const GHZ& benchmark) {
  return requestJSON("ghz", parametersJSON(benchmark)).dump();
}

std::string toRequestJSON(const Grover& benchmark) {
  return requestJSON("grover", parametersJSON(benchmark)).dump();
}

std::string toRequestJSON(const QPE& benchmark) {
  return requestJSON("qpe", parametersJSON(benchmark)).dump();
}

GHZ ghzFromManifestJSON(const std::string_view json,
                        const std::string_view source) {
  return parseManifest<GHZ>(json, source, "ghz", parseGHZParameters);
}

Grover groverFromManifestJSON(const std::string_view json,
                              const std::string_view source) {
  return parseManifest<Grover>(json, source, "grover", parseGroverParameters);
}

QPE qpeFromManifestJSON(const std::string_view json,
                        const std::string_view source) {
  return parseManifest<QPE>(json, source, "qpe", parseQPEParameters);
}

std::string toManifestJSON(const GHZ& benchmark) {
  return manifestJSON(benchmark).dump();
}

std::string toManifestJSON(const Grover& benchmark) {
  return manifestJSON(benchmark).dump();
}

std::string toManifestJSON(const QPE& benchmark) {
  return manifestJSON(benchmark).dump();
}

std::string caseId(const GHZ& benchmark) {
  return semanticCaseId(semanticJSON(benchmark));
}

std::string caseId(const Grover& benchmark) {
  return semanticCaseId(semanticJSON(benchmark));
}

std::string caseId(const QPE& benchmark) {
  return semanticCaseId(semanticJSON(benchmark));
}

Counts countsFromJSON(const std::string_view json,
                      const std::string_view source) {
  const auto root = parseJSON(json, source);
  requireObject(root, source, "$");
  rejectUnknownKeys(root, {"schema_version", "counts"}, source, "$");
  requireSchemaVersion(root, source);
  const auto& values = required(root, "counts", source, "$");
  requireObject(values, source, "$/counts");
  if (values.empty()) {
    fail(source, "$/counts", "must not be empty");
  }

  Counts result;
  size_t shots = 0;
  for (const auto& [outcome, countJSON] : values.items()) {
    if (outcome.empty() || !std::ranges::all_of(outcome, [](const char bit) {
          return bit == '0' || bit == '1';
        })) {
      fail(source, "$/counts", "outcomes must be non-empty bitstrings");
    }
    const auto pointer = "$/counts/" + outcome;
    const auto count = sizeValue(countJSON, source, pointer);
    if (count == 0) {
      fail(source, pointer, "must be positive");
    }
    if (count > std::numeric_limits<size_t>::max() - shots) {
      fail(source, "$/counts", "total shot count exceeds size_t");
    }
    shots += count;
    result.emplace(outcome, count);
  }
  return result;
}

std::string evaluationToJSON(const std::string_view caseIdValue,
                             const size_t shots, const Evaluation& evaluation) {
  if (!validCaseId(caseIdValue)) {
    throw std::invalid_argument("case ID must be a full lowercase SHA-256 ID");
  }
  if (shots == 0) {
    throw std::invalid_argument("evaluation requires at least one shot");
  }
  const auto validMetric = [](const double value) {
    return std::isfinite(value) && value >= 0. && value <= 1.;
  };
  if (!validMetric(evaluation.totalVariationDistance) ||
      !validMetric(evaluation.squaredHellingerFidelity) ||
      (evaluation.successProbability &&
       !validMetric(*evaluation.successProbability))) {
    throw std::invalid_argument(
        "evaluation metrics must be finite and in [0, 1]");
  }

  Json success = nullptr;
  if (evaluation.successProbability) {
    success = *evaluation.successProbability;
  }
  return Json{
      {"case_id", std::string(caseIdValue)},
      {"metrics",
       {{"squared_hellinger_fidelity", evaluation.squaredHellingerFidelity},
        {"success_probability", std::move(success)},
        {"total_variation_distance", evaluation.totalVariationDistance}}},
      {"schema_version", SCHEMA_VERSION},
      {"shots", shots}}
      .dump();
}

} // namespace mqt::benchmarks
