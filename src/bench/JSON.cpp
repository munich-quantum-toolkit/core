/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/JSON.hpp"

#include "bench/BV.hpp"
#include "bench/Error.hpp"
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

#include "JSON.hpp"
#include "SHA256.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bench {
namespace {

using Json = nlohmann::json;

constexpr uint64_t SCHEMA_VERSION = 1;
constexpr std::string_view CASE_DOMAIN = "mqt-core:benchmark-case:v1";

template <class Benchmark> struct BenchmarkMetadata;

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  template <> struct BenchmarkMetadata<TYPE> {                                 \
    static constexpr std::string_view id = ID;                                 \
    static constexpr uint64_t definitionVersion = DEFINITION_VERSION;          \
  };                                                                           \
  [[nodiscard]] Json STEM##InstanceSpecificationSchema();                      \
  [[nodiscard]] Result<std::string> evaluate##TYPE(std::string_view manifest,  \
                                                   std::string_view source,    \
                                                   const Counts& counts);
#include "bench/BenchmarkFamilies.inc"

using InstanceSpecificationSchemaFunction = Json (*)();
using EvaluationFunction = Result<std::string> (*)(std::string_view,
                                                   std::string_view,
                                                   const Counts&);

struct RegistryEntry {
  std::string_view id;
  uint64_t definitionVersion;
  InstanceSpecificationSchemaFunction instanceSpecificationSchema;
  EvaluationFunction evaluate;
  Result<BenchmarkInstance> (*parse)(std::string_view, std::string_view);
};
constexpr std::array REGISTRY{
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  RegistryEntry{                                                               \
      .id = (ID),                                                              \
      .definitionVersion = (DEFINITION_VERSION),                               \
      .instanceSpecificationSchema = STEM##InstanceSpecificationSchema,        \
      .evaluate = evaluate##TYPE,                                              \
      .parse = +[](std::string_view json,                                      \
                   std::string_view source) -> Result<BenchmarkInstance> {     \
        auto result = STEM##FromInstanceSpecificationJSON(json, source);       \
        if (auto* error = std::get_if<Error>(&result)) {                       \
          return std::move(*error);                                            \
        }                                                                      \
        return BenchmarkInstance{std::get<0>(std::move(result))};              \
      }},
#include "bench/BenchmarkFamilies.inc"
};
static_assert(
    [] {
      for (size_t index = 1; index < REGISTRY.size(); ++index) {
        if (REGISTRY[index - 1].id >= REGISTRY[index].id) {
          return false;
        }
      }
      return true;
    }(),
    "benchmark IDs must be unique and in lexical order");

[[nodiscard]] const RegistryEntry*
findBenchmark(const std::string_view benchmark) {
  for (const auto& entry : REGISTRY) {
    if (entry.id == benchmark) {
      return &entry;
    }
  }
  return nullptr;
}

[[nodiscard]] Error fail(const std::string_view source,
                         const std::string_view pointer,
                         const std::string_view message) {
  return Error{
      .message = std::string(source) + ":" + std::string(pointer) + " " +
                 std::string(message),
  };
}

template <class Benchmark>
[[nodiscard]] Result<Benchmark>
constructBenchmark(const std::string_view source, Result<Benchmark> result) {
  if (auto* error = std::get_if<Error>(&result)) {
    error->message = std::string(source) + ":$/parameters " + error->message;
  }
  return result;
}

[[nodiscard]] Result<Json> parseJSON(const std::string_view text,
                                     const std::string_view source) {
  std::vector<std::unordered_set<std::string>> keysByDepth;
  std::optional<Error> duplicate;
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
          if (!keysByDepth[index].emplace(key).second && !duplicate) {
            duplicate =
                fail(source, "$", "contains duplicate key '" + key + "'");
          }
        }
        return true;
      };
  auto result = mqt::detail::parseJSON(text, rejectDuplicates);
  if (duplicate) {
    return std::move(*duplicate);
  }
  if (auto const* error = std::get_if<std::string>(&result)) {
    return Error{.message = std::string(source) + ": invalid JSON: " + *error};
  }
  return std::get<0>(std::move(result));
}

std::optional<Error> requireObject(const Json& value,
                                   const std::string_view source,
                                   const std::string_view pointer) {
  if (!value.is_object()) {
    return fail(source, pointer, "must be an object");
  }
  return std::nullopt;
}

std::optional<Error> rejectUnknownKeys(
    const Json& value, const std::initializer_list<std::string_view> known,
    const std::string_view source, const std::string_view pointer) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(known, key) == known.end()) {
      return fail(source, pointer, "contains unknown key '" + key + "'");
    }
  }
  return std::nullopt;
}

[[nodiscard]] Result<const Json*> required(const Json& value,
                                           const char* const key,
                                           const std::string_view source,
                                           const std::string_view pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    return fail(source, std::string(pointer) + "/" + key, "is required");
  }
  return &*found;
}

[[nodiscard]] Result<uint64_t> unsignedInteger(const Json& value,
                                               const std::string_view source,
                                               const std::string_view pointer) {
  if (value.is_number_float()) {
    return fail(source, pointer, "must be encoded as an integer");
  }
  if (!value.is_number_unsigned() &&
      (!value.is_number_integer() || value.get<int64_t>() < 0)) {
    return fail(source, pointer, "must be a non-negative integer");
  }
  return value.get<uint64_t>();
}

[[nodiscard]] Result<size_t> sizeValue(const Json& value,
                                       const std::string_view source,
                                       const std::string_view pointer) {
  auto result = unsignedInteger(value, source, pointer);
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  const auto parsed = std::get<0>(result);
  if (parsed > std::numeric_limits<size_t>::max()) {
    return fail(source, pointer, "must fit size_t");
  }
  return static_cast<size_t>(parsed);
}

[[nodiscard]] Result<std::string> stringValue(const Json& value,
                                              const std::string_view source,
                                              const std::string_view pointer) {
  if (!value.is_string()) {
    return fail(source, pointer, "must be a string");
  }
  return value.get<std::string>();
}

std::optional<Error> requireSchemaVersion(const Json& root,
                                          const std::string_view source) {
  auto schemaVersionField = required(root, "schema_version", source, "$");
  if (auto* error = std::get_if<Error>(&schemaVersionField)) {
    return std::move(*error);
  }
  auto schemaVersionValue = unsignedInteger(*std::get<0>(schemaVersionField),
                                            source, "$/schema_version");
  if (auto* error = std::get_if<Error>(&schemaVersionValue)) {
    return std::move(*error);
  }
  const auto version = std::get<0>(schemaVersionValue);
  if (version != SCHEMA_VERSION) {
    return fail(source, "$/schema_version", "must be 1");
  }
  return std::nullopt;
}

[[nodiscard]] Result<const RegistryEntry*>
requireBenchmarkEntry(const Json& root, const std::string_view source) {
  auto benchmarkField = required(root, "benchmark", source, "$");
  if (auto* error = std::get_if<Error>(&benchmarkField)) {
    return std::move(*error);
  }
  auto benchmarkValue =
      stringValue(*std::get<0>(benchmarkField), source, "$/benchmark");
  if (auto* error = std::get_if<Error>(&benchmarkValue)) {
    return std::move(*error);
  }
  const auto benchmark = std::get<0>(benchmarkValue);
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry;
  }
  return fail(source, "$/benchmark",
              "selects unsupported benchmark '" + benchmark + "'");
}

[[nodiscard]] Result<Json> envelope(const std::string_view text,
                                    const std::string_view source,
                                    const bool manifest) {
  auto parsed = parseJSON(text, source);
  if (auto* error = std::get_if<Error>(&parsed)) {
    return std::move(*error);
  }
  auto& root = std::get<0>(parsed);
  if (auto error = requireObject(root, source, "$")) {
    return std::move(*error);
  }
  auto keyError = manifest
                      ? rejectUnknownKeys(root,
                                          {
                                              "schema_version",
                                              "case_id",
                                              "benchmark",
                                              "definition_version",
                                              "parameters",
                                              "outputs",
                                              "reference",
                                          },
                                          source, "$")
                      : rejectUnknownKeys(
                            root, {"schema_version", "benchmark", "parameters"},
                            source, "$");
  if (keyError) {
    return std::move(*keyError);
  }
  if (auto error = requireSchemaVersion(root, source)) {
    return std::move(*error);
  }
  auto entry = requireBenchmarkEntry(root, source);
  if (auto* error = std::get_if<Error>(&entry)) {
    return std::move(*error);
  }
  auto parameters = required(root, "parameters", source, "$");
  if (auto* error = std::get_if<Error>(&parameters)) {
    return std::move(*error);
  }
  if (auto error =
          requireObject(*std::get<0>(parameters), source, "$/parameters")) {
    return std::move(*error);
  }
  if (manifest) {
    auto definitionVersionField =
        required(root, "definition_version", source, "$");
    if (auto* error = std::get_if<Error>(&definitionVersionField)) {
      return std::move(*error);
    }
    auto definitionVersionValue = unsignedInteger(
        *std::get<0>(definitionVersionField), source, "$/definition_version");
    if (auto* error = std::get_if<Error>(&definitionVersionValue)) {
      return std::move(*error);
    }
    const auto definition = std::get<0>(definitionVersionValue);
    if (definition != std::get<0>(entry)->definitionVersion) {
      return fail(source, "$/definition_version",
                  "must be " +
                      std::to_string(std::get<0>(entry)->definitionVersion));
    }
    auto caseIdField = required(root, "case_id", source, "$");
    if (auto* error = std::get_if<Error>(&caseIdField)) {
      return std::move(*error);
    }
    auto caseIdValue =
        stringValue(*std::get<0>(caseIdField), source, "$/case_id");
    if (auto* error = std::get_if<Error>(&caseIdValue)) {
      return std::move(*error);
    }
    const auto caseId = std::get<0>(caseIdValue);
    static_cast<void>(caseId);
    auto outputsField = required(root, "outputs", source, "$");
    if (auto* error = std::get_if<Error>(&outputsField)) {
      return std::move(*error);
    }
    const auto& outputs = *std::get<0>(outputsField);
    if (!outputs.is_array()) {
      return fail(source, "$/outputs", "must be an array");
    }
    auto referenceField = required(root, "reference", source, "$");
    if (auto* error = std::get_if<Error>(&referenceField)) {
      return std::move(*error);
    }
    const auto& reference = *std::get<0>(referenceField);
    if (auto error = requireObject(reference, source, "$/reference")) {
      return std::move(*error);
    }
  }
  return std::move(root);
}

std::optional<Error> requireBenchmark(const Json& root,
                                      const std::string_view expected,
                                      const std::string_view source) {
  const auto& actual = root["benchmark"].get_ref<const std::string&>();
  if (actual != expected) {
    return fail(source, "$/benchmark",
                "must be '" + std::string(expected) + "'");
  }
  return std::nullopt;
}

[[nodiscard]] Result<BV> parseBVParameters(const Json& parameters,
                                           const std::string_view source) {
  if (auto error = rejectUnknownKeys(parameters, {"hidden_bitstring", "method"},
                                     source, "$/parameters")) {
    return std::move(*error);
  }
  auto hiddenBitstringField =
      required(parameters, "hidden_bitstring", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&hiddenBitstringField)) {
    return std::move(*error);
  }
  auto hiddenBitstringValue =
      stringValue(*std::get<0>(hiddenBitstringField), source,
                  "$/parameters/hidden_bitstring");
  if (auto* error = std::get_if<Error>(&hiddenBitstringValue)) {
    return std::move(*error);
  }
  BVOptions options{
      .hiddenBitstring = std::get<0>(hiddenBitstringValue),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    auto methodValue = stringValue(*method, source, "$/parameters/method");
    if (auto* error = std::get_if<Error>(&methodValue)) {
      return std::move(*error);
    }
    const auto value = std::get<0>(methodValue);
    if (value == "static") {
      options.method = BVMethod::Static;
    } else if (value == "dynamic") {
      options.method = BVMethod::Dynamic;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'static' or 'dynamic'");
    }
  }
  return constructBenchmark(source, BV::create(std::move(options)));
}

[[nodiscard]] Result<ModularMultiplier>
parseModularMultiplierParameters(const Json& parameters,
                                 const std::string_view source) {
  if (auto error = rejectUnknownKeys(
          parameters, {"multiplier", "modulus", "multiplicand", "control"},
          source, "$/parameters")) {
    return std::move(*error);
  }
  auto control = std::string("1");
  if (const auto value = parameters.find("control");
      value != parameters.end()) {
    auto controlValue = stringValue(*value, source, "$/parameters/control");
    if (auto* error = std::get_if<Error>(&controlValue)) {
      return std::move(*error);
    }
    control = std::get<0>(controlValue);
  }
  if (control.size() != 1U) {
    return fail(source, "$/parameters/control", "must be '0', '1', or '+'");
  }
  auto multiplicandField =
      required(parameters, "multiplicand", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&multiplicandField)) {
    return std::move(*error);
  }
  auto multiplicandValue = stringValue(*std::get<0>(multiplicandField), source,
                                       "$/parameters/multiplicand");
  if (auto* error = std::get_if<Error>(&multiplicandValue)) {
    return std::move(*error);
  }
  auto modulusField = required(parameters, "modulus", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&modulusField)) {
    return std::move(*error);
  }
  auto modulusValue =
      stringValue(*std::get<0>(modulusField), source, "$/parameters/modulus");
  if (auto* error = std::get_if<Error>(&modulusValue)) {
    return std::move(*error);
  }
  auto multiplierField =
      required(parameters, "multiplier", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&multiplierField)) {
    return std::move(*error);
  }
  auto multiplierValue = stringValue(*std::get<0>(multiplierField), source,
                                     "$/parameters/multiplier");
  if (auto* error = std::get_if<Error>(&multiplierValue)) {
    return std::move(*error);
  }
  return constructBenchmark(source,
                            ModularMultiplier::create({
                                .multiplier = std::get<0>(multiplierValue),
                                .modulus = std::get<0>(modulusValue),
                                .multiplicand = std::get<0>(multiplicandValue),
                                .control = control.front(),
                            }));
}

[[nodiscard]] Result<GHZ> parseGHZParameters(const Json& parameters,
                                             const std::string_view source) {
  if (auto error =
          rejectUnknownKeys(parameters, {"qubits", "topology", "basis"}, source,
                            "$/parameters")) {
    return std::move(*error);
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&qubitsField)) {
    return std::move(*error);
  }
  auto qubitsValue =
      sizeValue(*std::get<0>(qubitsField), source, "$/parameters/qubits");
  if (auto* error = std::get_if<Error>(&qubitsValue)) {
    return std::move(*error);
  }
  GHZOptions options{
      .qubits = std::get<0>(qubitsValue),
  };
  if (const auto topology = parameters.find("topology");
      topology != parameters.end()) {
    auto topologyValue =
        stringValue(*topology, source, "$/parameters/topology");
    if (auto* error = std::get_if<Error>(&topologyValue)) {
      return std::move(*error);
    }
    const auto value = std::get<0>(topologyValue);
    if (value == "linear") {
      options.topology = GHZTopology::Linear;
    } else if (value == "star") {
      options.topology = GHZTopology::Star;
    } else {
      return fail(source, "$/parameters/topology",
                  "must be 'linear' or 'star'");
    }
  }
  if (const auto basis = parameters.find("basis"); basis != parameters.end()) {
    auto basisValue = stringValue(*basis, source, "$/parameters/basis");
    if (auto* error = std::get_if<Error>(&basisValue)) {
      return std::move(*error);
    }
    const auto value = std::get<0>(basisValue);
    if (value == "z") {
      options.basis = GHZBasis::Z;
    } else if (value == "x") {
      options.basis = GHZBasis::X;
    } else {
      return fail(source, "$/parameters/basis", "must be 'z' or 'x'");
    }
  }
  return constructBenchmark(source, GHZ::create(options));
}

[[nodiscard]] Result<Grover>
parseGroverParameters(const Json& parameters, const std::string_view source) {
  if (auto error =
          rejectUnknownKeys(parameters, {"marked_bitstring", "iterations"},
                            source, "$/parameters")) {
    return std::move(*error);
  }
  auto markedBitstringField =
      required(parameters, "marked_bitstring", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&markedBitstringField)) {
    return std::move(*error);
  }
  auto markedBitstringValue =
      stringValue(*std::get<0>(markedBitstringField), source,
                  "$/parameters/marked_bitstring");
  if (auto* error = std::get_if<Error>(&markedBitstringValue)) {
    return std::move(*error);
  }
  GroverOptions options{
      .markedBitstring = std::get<0>(markedBitstringValue),
  };
  if (const auto iterations = parameters.find("iterations");
      iterations != parameters.end()) {
    auto iterationsValue =
        sizeValue(*iterations, source, "$/parameters/iterations");
    if (auto* error = std::get_if<Error>(&iterationsValue)) {
      return std::move(*error);
    }
    options.iterations = std::get<0>(iterationsValue);
  }
  return constructBenchmark(source, Grover::create(std::move(options)));
}

[[nodiscard]] Result<Multiplexer>
parseMultiplexerParameters(const Json& parameters,
                           const std::string_view source) {
  if (auto error =
          rejectUnknownKeys(parameters, {"qubits"}, source, "$/parameters")) {
    return std::move(*error);
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&qubitsField)) {
    return std::move(*error);
  }
  auto qubitsValue =
      sizeValue(*std::get<0>(qubitsField), source, "$/parameters/qubits");
  if (auto* error = std::get_if<Error>(&qubitsValue)) {
    return std::move(*error);
  }
  return constructBenchmark(source, Multiplexer::create({
                                        .qubits = std::get<0>(qubitsValue),
                                    }));
}

[[nodiscard]] Result<QFT> parseQFTParameters(const Json& parameters,
                                             const std::string_view source) {
  if (auto error =
          rejectUnknownKeys(parameters, {"qubits", "period_exponent", "method"},
                            source, "$/parameters")) {
    return std::move(*error);
  }
  auto periodExponentField =
      required(parameters, "period_exponent", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&periodExponentField)) {
    return std::move(*error);
  }
  auto periodExponentValue = sizeValue(*std::get<0>(periodExponentField),
                                       source, "$/parameters/period_exponent");
  if (auto* error = std::get_if<Error>(&periodExponentValue)) {
    return std::move(*error);
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&qubitsField)) {
    return std::move(*error);
  }
  auto qubitsValue =
      sizeValue(*std::get<0>(qubitsField), source, "$/parameters/qubits");
  if (auto* error = std::get_if<Error>(&qubitsValue)) {
    return std::move(*error);
  }
  QFTOptions options{
      .qubits = std::get<0>(qubitsValue),
      .periodExponent = std::get<0>(periodExponentValue),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    auto methodValue = stringValue(*method, source, "$/parameters/method");
    if (auto* error = std::get_if<Error>(&methodValue)) {
      return std::move(*error);
    }
    const auto value = std::get<0>(methodValue);
    if (value == "standard") {
      options.method = QFTMethod::Standard;
    } else if (value == "semiclassical") {
      options.method = QFTMethod::Semiclassical;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'standard' or 'semiclassical'");
    }
  }
  return constructBenchmark(source, QFT::create(options));
}

[[nodiscard]] Result<QFTAdder>
parseQFTAdderParameters(const Json& parameters, const std::string_view source) {
  if (auto error = rejectUnknownKeys(
          parameters, {"addend", "accumulator", "method", "overflow"}, source,
          "$/parameters")) {
    return std::move(*error);
  }
  auto accumulatorField =
      required(parameters, "accumulator", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&accumulatorField)) {
    return std::move(*error);
  }
  auto accumulatorValue = stringValue(*std::get<0>(accumulatorField), source,
                                      "$/parameters/accumulator");
  if (auto* error = std::get_if<Error>(&accumulatorValue)) {
    return std::move(*error);
  }
  auto addendField = required(parameters, "addend", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&addendField)) {
    return std::move(*error);
  }
  auto addendValue =
      stringValue(*std::get<0>(addendField), source, "$/parameters/addend");
  if (auto* error = std::get_if<Error>(&addendValue)) {
    return std::move(*error);
  }
  QFTAdderOptions options{
      .addend = std::get<0>(addendValue),
      .accumulator = std::get<0>(accumulatorValue),
  };
  if (const auto it = parameters.find("method"); it != parameters.end()) {
    auto methodValue = stringValue(*it, source, "$/parameters/method");
    if (auto* error = std::get_if<Error>(&methodValue)) {
      return std::move(*error);
    }
    const auto value = std::get<0>(methodValue);
    if (value == "register") {
      options.method = QFTAdderMethod::Register;
    } else if (value == "constant") {
      options.method = QFTAdderMethod::Constant;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'register' or 'constant'");
    }
  }
  if (const auto it = parameters.find("overflow"); it != parameters.end()) {
    auto overflowValue = stringValue(*it, source, "$/parameters/overflow");
    if (auto* error = std::get_if<Error>(&overflowValue)) {
      return std::move(*error);
    }
    const auto value = std::get<0>(overflowValue);
    if (value == "wrap") {
      options.overflow = QFTAdderOverflow::Wrap;
    } else if (value == "carry") {
      options.overflow = QFTAdderOverflow::Carry;
    } else {
      return fail(source, "$/parameters/overflow", "must be 'wrap' or 'carry'");
    }
  }
  return constructBenchmark(source, QFTAdder::create(std::move(options)));
}

[[nodiscard]] Result<QPE> parseQPEParameters(const Json& parameters,
                                             const std::string_view source) {
  if (auto error =
          rejectUnknownKeys(parameters, {"precision", "phase", "method"},
                            source, "$/parameters")) {
    return std::move(*error);
  }
  auto precisionField =
      required(parameters, "precision", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&precisionField)) {
    return std::move(*error);
  }
  auto precisionValue =
      sizeValue(*std::get<0>(precisionField), source, "$/parameters/precision");
  if (auto* error = std::get_if<Error>(&precisionValue)) {
    return std::move(*error);
  }
  const auto precision = std::get<0>(precisionValue);
  auto phaseField = required(parameters, "phase", source, "$/parameters");
  if (auto* error = std::get_if<Error>(&phaseField)) {
    return std::move(*error);
  }
  const auto& phase = *std::get<0>(phaseField);
  if (auto error = requireObject(phase, source, "$/parameters/phase")) {
    return std::move(*error);
  }
  if (auto error = rejectUnknownKeys(phase, {"numerator", "denominator"},
                                     source, "$/parameters/phase")) {
    return std::move(*error);
  }
  auto numeratorField =
      required(phase, "numerator", source, "$/parameters/phase");
  if (auto* error = std::get_if<Error>(&numeratorField)) {
    return std::move(*error);
  }
  auto numeratorValue = unsignedInteger(*std::get<0>(numeratorField), source,
                                        "$/parameters/phase/numerator");
  if (auto* error = std::get_if<Error>(&numeratorValue)) {
    return std::move(*error);
  }
  const auto numerator = std::get<0>(numeratorValue);
  auto denominatorField =
      required(phase, "denominator", source, "$/parameters/phase");
  if (auto* error = std::get_if<Error>(&denominatorField)) {
    return std::move(*error);
  }
  auto denominatorValue = unsignedInteger(
      *std::get<0>(denominatorField), source, "$/parameters/phase/denominator");
  if (auto* error = std::get_if<Error>(&denominatorValue)) {
    return std::move(*error);
  }
  const auto denominator = std::get<0>(denominatorValue);
  auto method = QPEMethod::Standard;
  if (const auto value = parameters.find("method"); value != parameters.end()) {
    auto methodValue = stringValue(*value, source, "$/parameters/method");
    if (auto* error = std::get_if<Error>(&methodValue)) {
      return std::move(*error);
    }
    const auto name = std::get<0>(methodValue);
    if (name == "standard") {
      method = QPEMethod::Standard;
    } else if (name == "iterative") {
      method = QPEMethod::Iterative;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'standard' or 'iterative'");
    }
  }
  auto phaseValue =
      constructBenchmark(source, Phase::create(numerator, denominator));
  if (auto* error = std::get_if<Error>(&phaseValue)) {
    return std::move(*error);
  }
  return constructBenchmark(source, QPE::create({
                                        .precision = precision,
                                        .phase = std::get<0>(phaseValue),
                                        .method = method,
                                    }));
}

[[nodiscard]] Result<RepeatUntilSuccess>
parseRepeatUntilSuccessParameters(const Json& parameters,
                                  const std::string_view source) {
  if (auto error = rejectUnknownKeys(parameters, {"data_qubits"}, source,
                                     "$/parameters")) {
    return std::move(*error);
  }
  RepeatUntilSuccessOptions options;
  if (const auto width = parameters.find("data_qubits");
      width != parameters.end()) {
    auto dataQubitsValue =
        sizeValue(*width, source, "$/parameters/data_qubits");
    if (auto* error = std::get_if<Error>(&dataQubitsValue)) {
      return std::move(*error);
    }
    options.dataQubits = std::get<0>(dataQubitsValue);
  }
  return constructBenchmark(source, RepeatUntilSuccess::create(options));
}

[[nodiscard]] Result<Teleportation>
parseTeleportationParameters(const Json& parameters,
                             const std::string_view source) {
  if (auto error = rejectUnknownKeys(parameters, {}, source, "$/parameters")) {
    return std::move(*error);
  }
  return Teleportation{};
}

[[nodiscard]] std::string_view topologyName(const GHZTopology topology) {
  return topology == GHZTopology::Linear ? "linear" : "star";
}

[[nodiscard]] std::string_view basisName(const GHZBasis basis) {
  return basis == GHZBasis::Z ? "z" : "x";
}

[[nodiscard]] std::string methodName(const BVMethod method) {
  return method == BVMethod::Static ? "static" : "dynamic";
}

[[nodiscard]] std::string methodName(const QFTMethod method) {
  return method == QFTMethod::Standard ? "standard" : "semiclassical";
}

[[nodiscard]] std::string methodName(const QPEMethod method) {
  return method == QPEMethod::Standard ? "standard" : "iterative";
}

[[nodiscard]] Json parametersJSON(const BV& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"hidden_bitstring", options.hiddenBitstring},
      {"method", methodName(options.method)},
  };
}

[[nodiscard]] Json parametersJSON(const ModularMultiplier& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"control", std::string(1, options.control)},
      {"multiplicand", options.multiplicand},
      {"modulus", options.modulus},
      {"multiplier", options.multiplier},
  };
}

[[nodiscard]] Json parametersJSON(const GHZ& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"basis", basisName(options.basis)},
      {"qubits", options.qubits},
      {"topology", topologyName(options.topology)},
  };
}

[[nodiscard]] Json parametersJSON(const Grover& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"iterations", *options.iterations},
      {"marked_bitstring", options.markedBitstring},
  };
}

[[nodiscard]] Json parametersJSON(const Multiplexer& benchmark) {
  return {{"qubits", benchmark.options().qubits}};
}

[[nodiscard]] Json parametersJSON(const QFT& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"method", methodName(options.method)},
      {"period_exponent", options.periodExponent},
      {"qubits", options.qubits},
  };
}

[[nodiscard]] Json parametersJSON(const QFTAdder& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"addend", options.addend},
      {"accumulator", options.accumulator},
      {
          "method",
          options.method == QFTAdderMethod::Register ? "register" : "constant",
      },
      {
          "overflow",
          options.overflow == QFTAdderOverflow::Wrap ? "wrap" : "carry",
      },
  };
}

[[nodiscard]] Json parametersJSON(const QPE& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"method", methodName(options.method)},
      {
          "phase",
          {
              {"denominator", options.phase.denominator()},
              {"numerator", options.phase.numerator()},
          },
      },
      {"precision", options.precision},
  };
}

[[nodiscard]] Json parametersJSON(const RepeatUntilSuccess& benchmark) {
  return {{"data_qubits", benchmark.options().dataQubits}};
}

[[nodiscard]] Json parametersJSON(const Teleportation& /*unused*/) {
  return Json::object();
}

[[nodiscard]] Json analyticReferenceJSON(
    const Output& output, const std::string_view model,
    const std::optional<std::string_view> successOutcome = std::nullopt) {
  Json reference = {
      {"kind", "analytic"},
      {"model", std::string(model)},
      {"outcome_order", "big_endian"},
      {"output", output.name},
      {"version", 1},
  };
  if (successOutcome) {
    reference["success_outcome"] = std::string(*successOutcome);
  }
  return reference;
}

[[nodiscard]] Json referenceJSON(const BV& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "bernstein_vazirani",
                               benchmark.options().hiddenBitstring);
}

[[nodiscard]] Json referenceJSON(const ModularMultiplier& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "modular_multiplier",
                               benchmark.expectedResult());
}

[[nodiscard]] Json referenceJSON(const GHZ& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "ghz");
}

[[nodiscard]] Json referenceJSON(const Grover& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "grover_single_marked",
                               benchmark.options().markedBitstring);
}

[[nodiscard]] Json referenceJSON(const Multiplexer& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "multiplexer");
}

[[nodiscard]] Json referenceJSON(const QFT& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "qft_power_of_two_period");
}

[[nodiscard]] Json referenceJSON(const QFTAdder& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "qft_adder",
                               benchmark.expectedResult());
}

[[nodiscard]] Json referenceJSON(const QPE& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "qpe_dirichlet");
}

[[nodiscard]] Json referenceJSON(const RepeatUntilSuccess& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "repeat_until_success");
}

[[nodiscard]] Json referenceJSON(const Teleportation& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "teleportation", "0");
}

[[nodiscard]] Json semanticJSON(const std::string_view id,
                                const uint64_t definitionVersionValue,
                                const Json& parameters, const Output& output,
                                const Json& reference) {
  return {
      {"benchmark", std::string(id)},
      {"definition_version", definitionVersionValue},
      {
          "outputs",
          Json::array({{{"name", output.name}, {"width", output.width}}}),
      },
      {"parameters", parameters},
      {"reference", reference},
  };
}

template <class Benchmark>
[[nodiscard]] Json semanticJSON(const Benchmark& benchmark) {
  using Metadata = BenchmarkMetadata<Benchmark>;
  return semanticJSON(Metadata::id, Metadata::definitionVersion,
                      parametersJSON(benchmark), benchmark.output(),
                      referenceJSON(benchmark));
}

[[nodiscard]] Result<std::string> semanticCaseId(const Json& semantic) {
  auto input = std::string(CASE_DOMAIN);
  input.push_back('\0');
  input += semantic.dump();
  auto digest = detail::sha256Hex(input);
  if (auto* error = std::get_if<Error>(&digest)) {
    return std::move(*error);
  }
  return "sha256-" + std::get<0>(digest);
}

template <class Benchmark>
[[nodiscard]] Result<Json> manifestJSON(const Benchmark& benchmark) {
  auto semantic = semanticJSON(benchmark);
  auto id = semanticCaseId(semantic);
  if (auto* error = std::get_if<Error>(&id)) {
    return std::move(*error);
  }
  semantic["case_id"] = std::get<0>(std::move(id));
  semantic["schema_version"] = SCHEMA_VERSION;
  return semantic;
}

template <class Benchmark>
[[nodiscard]] Json instanceSpecificationJSON(const Benchmark& benchmark) {
  return {
      {"benchmark", std::string(BenchmarkMetadata<Benchmark>::id)},
      {"parameters", parametersJSON(benchmark)},
      {"schema_version", SCHEMA_VERSION},
  };
}

template <class Benchmark, class ParseParameters>
[[nodiscard]] Result<Benchmark>
parseBenchmark(const std::string_view text, const std::string_view source,
               const ParseParameters& parseParameters, const bool manifest) {
  auto parsed = envelope(text, source, manifest);
  if (auto* error = std::get_if<Error>(&parsed)) {
    return std::move(*error);
  }
  const auto& root = std::get<0>(parsed);
  if (auto error =
          requireBenchmark(root, BenchmarkMetadata<Benchmark>::id, source)) {
    return std::move(*error);
  }
  auto benchmark = parseParameters(root["parameters"], source);
  if (auto* error = std::get_if<Error>(&benchmark)) {
    return std::move(*error);
  }
  if (manifest) {
    auto expected = manifestJSON(std::get<0>(benchmark));
    if (auto* error = std::get_if<Error>(&expected)) {
      return std::move(*error);
    }
    if (root.dump() != std::get<0>(expected).dump()) {
      return fail(source, "$",
                  "does not match its resolved benchmark instance and case ID");
    }
  }
  return benchmark;
}

template <class Benchmark>
[[nodiscard]] Json baseInstanceSpecificationSchema(Json parameters) {
  using Metadata = BenchmarkMetadata<Benchmark>;
  return {
      {"$schema", "https://json-schema.org/draft/2020-12/schema"},
      {"additionalProperties", false},
      {
          "properties",
          {
              {"benchmark", {{"const", std::string(Metadata::id)}}},
              {"parameters", std::move(parameters)},
              {"schema_version", {{"const", SCHEMA_VERSION}}},
          },
      },
      {"required", {"schema_version", "benchmark", "parameters"}},
      {"type", "object"},
      {"x-mqt-definition-version", Metadata::definitionVersion},
  };
}

[[nodiscard]] Json bvInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<BV>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "hidden_bitstring",
                  {
                      {"maxLength", BVOptions::MAX_BITS},
                      {"minLength", 1},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
              {
                  "method",
                  {{"default", "static"}, {"enum", {"static", "dynamic"}}},
              },
          },
      },
      {"required", {"hidden_bitstring"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json modularMultiplierInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<ModularMultiplier>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {"control", {{"default", "1"}, {"enum", {"0", "1", "+"}}}},
              {
                  "multiplicand",
                  {
                      {"type", "string"},
                      {"minLength", 2},
                      {"maxLength", ModularMultiplierOptions::MAX_BITS},
                      {"pattern", "^[01+]+$"},
                  },
              },
              {
                  "modulus",
                  {
                      {
                          "maxLength",
                          ModularMultiplierOptions::MAX_BITS,
                      },
                      {"minLength", 2},
                      {"pattern", "^1[01]+$"},
                      {"type", "string"},
                  },
              },
              {
                  "multiplier",
                  {
                      {
                          "maxLength",
                          ModularMultiplierOptions::MAX_BITS,
                      },
                      {"minLength", 2},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
          },
      },
      {"required", {"multiplier", "modulus", "multiplicand"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json ghzInstanceSpecificationSchema() {
  Json parameters{
      {"additionalProperties", false},
      {
          "properties",
          {
              {"basis", {{"default", "z"}, {"enum", {"z", "x"}}}},
              {
                  "qubits",
                  {
                      {"maximum", GHZOptions::MAX_QUBITS},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
              {
                  "topology",
                  {{"default", "linear"}, {"enum", {"linear", "star"}}},
              },
          },
      },
      {"required", {"qubits"}},
      {"type", "object"},
  };
  parameters["allOf"] = Json::array({
      {
          {
              "if",
              {
                  {"properties", {{"basis", {{"const", "x"}}}}},
                  {"required", {"basis"}},
              },
          },
          {
              "then",
              {
                  {
                      "properties",
                      {
                          {
                              "qubits",
                              {{"maximum", GHZOptions::MAX_X_BASIS_QUBITS}},
                          },
                      },
                  },
              },
          },
      },
  });
  return baseInstanceSpecificationSchema<GHZ>(std::move(parameters));
}

[[nodiscard]] Json groverInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Grover>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "iterations",
                  {
                      {"maximum", std::numeric_limits<int32_t>::max()},
                      {"minimum", 0},
                      {"type", "integer"},
                  },
              },
              {
                  "marked_bitstring",
                  {
                      {"maxLength", 62},
                      {"minLength", 2},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
          },
      },
      {"required", {"marked_bitstring"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json multiplexerInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Multiplexer>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "qubits",
                  {
                      {"maximum", MultiplexerOptions::MAX_QUBITS},
                      {"minimum", 2},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", {"qubits"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json qftInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<QFT>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "method",
                  {
                      {"default", "standard"},
                      {"enum", {"standard", "semiclassical"}},
                  },
              },
              {
                  "period_exponent",
                  {
                      {"maximum", QFTOptions::MAX_PERIOD_EXPONENT},
                      {"minimum", 0},
                      {"type", "integer"},
                  },
              },
              {
                  "qubits",
                  {
                      {"maximum", QFTOptions::MAX_QUBITS},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", {"qubits", "period_exponent"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json qftAdderInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<QFTAdder>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "addend",
                  {
                      {"type", "string"},
                      {"minLength", 1},
                      {"maxLength", QFTAdderOptions::MAX_QUBITS},
                      {"pattern", "^[01+]+$"},
                  },
              },
              {
                  "accumulator",
                  {
                      {"type", "string"},
                      {"minLength", 1},
                      {"maxLength", QFTAdderOptions::MAX_QUBITS},
                      {"pattern", "^[01]+$"},
                  },
              },
              {
                  "method",
                  {
                      {"type", "string"},
                      {"enum", {"register", "constant"}},
                      {"default", "register"},
                  },
              },
              {
                  "overflow",
                  {
                      {"type", "string"},
                      {"enum", {"wrap", "carry"}},
                      {"default", "wrap"},
                  },
              },
          },
      },
      {
          "allOf",
          {
              {
                  {
                      "if",
                      {
                          {"properties", {{"method", {{"const", "constant"}}}}},
                          {"required", {"method"}},
                      },
                  },
                  {
                      "then",
                      {{"properties", {{"addend", {{"pattern", "^[01]+$"}}}}}},
                  },
              },
              {
                  {
                      "if",
                      {
                          {"properties", {{"overflow", {{"const", "carry"}}}}},
                          {"required", {"overflow"}},
                      },
                  },
                  {
                      "then",
                      {
                          {
                              "properties",
                              {
                                  {
                                      "addend",
                                      {
                                          {
                                              "maxLength",
                                              QFTAdderOptions::MAX_QUBITS - 1U,
                                          },
                                      },
                                  },
                                  {
                                      "accumulator",
                                      {
                                          {
                                              "maxLength",
                                              QFTAdderOptions::MAX_QUBITS - 1U,
                                          },
                                      },
                                  },
                              },
                          },
                      },
                  },
              },
          },
      },
      {"required", {"addend", "accumulator"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json qpeInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<QPE>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "method",
                  {
                      {"default", "standard"},
                      {"enum", {"standard", "iterative"}},
                  },
              },
              {
                  "phase",
                  {
                      {"additionalProperties", false},
                      {
                          "properties",
                          {
                              {
                                  "denominator",
                                  {
                                      {
                                          "maximum",
                                          std::numeric_limits<uint64_t>::max(),
                                      },
                                      {"minimum", 1},
                                      {"type", "integer"},
                                  },
                              },
                              {
                                  "numerator",
                                  {
                                      {
                                          "maximum",
                                          std::numeric_limits<uint64_t>::max(),
                                      },
                                      {"minimum", 0},
                                      {"type", "integer"},
                                  },
                              },
                          },
                      },
                      {"required", {"numerator", "denominator"}},
                      {"type", "object"},
                  },
              },
              {
                  "precision",
                  {
                      {"maximum", QPEOptions::MAX_PRECISION},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", {"precision", "phase"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json repeatUntilSuccessInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<RepeatUntilSuccess>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "data_qubits",
                  {
                      {"default", 1},
                      {"minimum", 1},
                      {"maximum", RepeatUntilSuccessOptions::MAX_DATA_QUBITS},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"type", "object"},
  });
}

[[nodiscard]] Json teleportationInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Teleportation>({
      {"additionalProperties", false},
      {"properties", Json::object()},
      {"type", "object"},
  });
}

template <class Benchmark>
[[nodiscard]] Result<std::string> evaluateBenchmark(const Benchmark& benchmark,
                                                    const Counts& counts) {
  auto evaluation = benchmark.evaluate(counts);
  if (auto* error = std::get_if<Error>(&evaluation)) {
    return std::move(*error);
  }
  auto id = caseId(benchmark);
  if (auto* error = std::get_if<Error>(&id)) {
    return std::move(*error);
  }
  /// Evaluation validates the total before this sum.
  const auto shots = std::accumulate(
      counts.begin(), counts.end(), size_t{0},
      [](const size_t sum, const auto& item) { return sum + item.second; });
  return evaluationToJSON(std::get<0>(id), shots, std::get<0>(evaluation));
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  Result<std::string> evaluate##TYPE(const std::string_view manifest,          \
                                     const std::string_view source,            \
                                     const Counts& counts) {                   \
    auto result = STEM##FromManifestJSON(manifest, source);                    \
    if (auto* error = std::get_if<Error>(&result)) {                           \
      return std::move(*error);                                                \
    }                                                                          \
    return evaluateBenchmark(std::get<0>(result), counts);                     \
  }
#include "bench/BenchmarkFamilies.inc"

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

Result<std::string>
benchmarkIdFromInstanceSpecificationJSON(const std::string_view json,
                                         const std::string_view source) {
  auto result = envelope(json, source, false);
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  return std::get<0>(result)["benchmark"].get<std::string>();
}

Result<std::string> benchmarkIdFromManifestJSON(const std::string_view json,
                                                const std::string_view source) {
  auto result = envelope(json, source, true);
  if (auto* error = std::get_if<Error>(&result)) {
    return std::move(*error);
  }
  return std::get<0>(result)["benchmark"].get<std::string>();
}

std::string listBenchmarksJSON() {
  auto benchmarks = Json::array();
  for (const auto& entry : REGISTRY) {
    benchmarks.emplace_back(Json{
        {"definition_version", entry.definitionVersion},
        {"id", std::string(entry.id)},
    });
  }
  return Json{
      {"benchmarks", std::move(benchmarks)},
      {"schema_version", SCHEMA_VERSION},
  }
      .dump();
}

Result<std::string> describeBenchmarkJSON(const std::string_view benchmark) {
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry->instanceSpecificationSchema().dump();
  }
  return Error{
      .message = "unsupported benchmark '" + std::string(benchmark) + "'",
  };
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  Result<TYPE> STEM##FromInstanceSpecificationJSON(                            \
      const std::string_view json, const std::string_view source) {            \
    return parseBenchmark<TYPE>(json, source, parse##TYPE##Parameters, false); \
  }                                                                            \
  std::string toInstanceSpecificationJSON(const TYPE& benchmark) {             \
    return instanceSpecificationJSON(benchmark).dump();                        \
  }                                                                            \
  Result<TYPE> STEM##FromManifestJSON(const std::string_view json,             \
                                      const std::string_view source) {         \
    return parseBenchmark<TYPE>(json, source, parse##TYPE##Parameters, true);  \
  }                                                                            \
  Result<std::string> toManifestJSON(const TYPE& benchmark) {                  \
    auto result = manifestJSON(benchmark);                                     \
    if (auto* error = std::get_if<Error>(&result)) {                           \
      return std::move(*error);                                                \
    }                                                                          \
    return std::get<0>(result).dump();                                         \
  }                                                                            \
  Result<std::string> caseId(const TYPE& benchmark) {                          \
    return semanticCaseId(semanticJSON(benchmark));                            \
  }
#include "bench/BenchmarkFamilies.inc"

Result<Counts> countsFromJSON(const std::string_view json,
                              const std::string_view source) {
  auto parsed = parseJSON(json, source);
  if (auto* error = std::get_if<Error>(&parsed)) {
    return std::move(*error);
  }
  const auto& root = std::get<0>(parsed);
  if (auto error = requireObject(root, source, "$")) {
    return std::move(*error);
  }
  if (auto error =
          rejectUnknownKeys(root, {"schema_version", "counts"}, source, "$")) {
    return std::move(*error);
  }
  if (auto error = requireSchemaVersion(root, source)) {
    return std::move(*error);
  }
  auto field = required(root, "counts", source, "$");
  if (auto* error = std::get_if<Error>(&field)) {
    return std::move(*error);
  }
  const auto& values = *std::get<0>(field);
  if (auto error = requireObject(values, source, "$/counts")) {
    return std::move(*error);
  }
  if (values.empty()) {
    return fail(source, "$/counts", "must not be empty");
  }

  Counts result;
  size_t shots = 0;
  for (const auto& [outcome, countJSON] : values.items()) {
    if (outcome.empty() || !std::ranges::all_of(outcome, [](const char bit) {
          return bit == '0' || bit == '1';
        })) {
      return fail(source, "$/counts", "outcomes must be non-empty bitstrings");
    }
    const auto pointer = "$/counts/" + outcome;
    auto countResult = sizeValue(countJSON, source, pointer);
    if (auto* error = std::get_if<Error>(&countResult)) {
      return std::move(*error);
    }
    const auto count = std::get<0>(countResult);
    if (count == 0) {
      return fail(source, pointer, "must be positive");
    }
    if (count > std::numeric_limits<size_t>::max() - shots) {
      return fail(source, "$/counts", "total shot count exceeds size_t");
    }
    shots += count;
    result.emplace(outcome, count);
  }
  return result;
}

Result<std::string> evaluateJSON(const std::string_view manifest,
                                 const std::string_view counts,
                                 const std::string_view manifestSource,
                                 const std::string_view countsSource) {
  auto id = benchmarkIdFromManifestJSON(manifest, manifestSource);
  if (auto* error = std::get_if<Error>(&id)) {
    return std::move(*error);
  }
  auto parsedCounts = countsFromJSON(counts, countsSource);
  if (auto* error = std::get_if<Error>(&parsedCounts)) {
    return std::move(*error);
  }
  return findBenchmark(std::get<0>(id))
      ->evaluate(manifest, manifestSource, std::get<0>(parsedCounts));
}

Result<std::string> evaluationToJSON(const std::string_view caseIdValue,
                                     const size_t shots,
                                     const Evaluation& evaluation) {
  if (!validCaseId(caseIdValue)) {
    return Error{.message = "case ID must be a full lowercase SHA-256 ID"};
  }
  if (shots == 0) {
    return Error{.message = "evaluation requires at least one shot"};
  }
  const auto validMetric = [](const double value) {
    return std::isfinite(value) && value >= 0. && value <= 1.;
  };
  if (!validMetric(evaluation.totalVariationDistance) ||
      !validMetric(evaluation.squaredHellingerFidelity) ||
      (evaluation.successProbability &&
       !validMetric(*evaluation.successProbability))) {
    return Error{.message = "evaluation metrics must be finite and in [0, 1]"};
  }

  Json success = nullptr;
  if (evaluation.successProbability) {
    success = *evaluation.successProbability;
  }
  return Json{
      {"case_id", std::string(caseIdValue)},
      {
          "metrics",
          {
              {
                  "squared_hellinger_fidelity",
                  evaluation.squaredHellingerFidelity,
              },
              {"success_probability", std::move(success)},
              {"total_variation_distance", evaluation.totalVariationDistance},
          },
      },
      {"schema_version", SCHEMA_VERSION},
      {"shots", shots},
  }
      .dump();
}

Result<ParsedBenchmark>
parseInstanceSpecificationJSON(const std::string_view json,
                               const std::string_view source) {
  auto id = benchmarkIdFromInstanceSpecificationJSON(json, source);
  if (auto* error = std::get_if<Error>(&id)) {
    return std::move(*error);
  }
  auto instance = findBenchmark(std::get<0>(id))->parse(json, source);
  if (auto* error = std::get_if<Error>(&instance)) {
    return std::move(*error);
  }
  return std::visit(
      [&](auto&& benchmark) -> Result<ParsedBenchmark> {
        auto caseIdValue = caseId(benchmark);
        if (auto* error = std::get_if<Error>(&caseIdValue)) {
          return std::move(*error);
        }
        auto manifest = toManifestJSON(benchmark);
        if (auto* error = std::get_if<Error>(&manifest)) {
          return std::move(*error);
        }
        return ParsedBenchmark{
            .instance = std::forward<decltype(benchmark)>(benchmark),
            .benchmarkId = std::get<0>(std::move(id)),
            .caseId = std::get<0>(std::move(caseIdValue)),
            .manifestJSON = std::get<0>(std::move(manifest)),
        };
      },
      std::get<0>(std::move(instance)));
}

} // namespace mqt::bench
