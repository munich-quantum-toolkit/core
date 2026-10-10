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
#include "bench/Evaluation.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/MagicStateDistillation.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Shor.hpp"
#include "bench/Teleportation.hpp"
#include "bench/WState.hpp"
#include "bench/WeakMeasurementGrover.hpp"

#include "JSON.hpp"
#include "support/Diagnostics.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SHA256.h"

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
  [[nodiscard]] llvm::FailureOr<TYPE> parse##TYPE##Parameters(                 \
      const Json& parameters, std::string_view source);
#include "bench/BenchmarkFamilies.inc"

using InstanceSpecificationSchemaFunction = Json (*)();

struct RegistryEntry {
  std::string_view id;
  uint64_t definitionVersion;
  InstanceSpecificationSchemaFunction instanceSpecificationSchema;
  llvm::FailureOr<BenchmarkInstance> (*parse)(const Json&, std::string_view);
};
constexpr std::array REGISTRY{
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  RegistryEntry{.id = (ID),                                                    \
                .definitionVersion = (DEFINITION_VERSION),                     \
                .instanceSpecificationSchema =                                 \
                    STEM##InstanceSpecificationSchema,                         \
                .parse = +[](const Json& parameters, std::string_view source)  \
                    -> llvm::FailureOr<BenchmarkInstance> {                    \
                  auto result = parse##TYPE##Parameters(parameters, source);   \
                  if (llvm::failed(result)) {                                  \
                    return llvm::failure();                                    \
                  }                                                            \
                  return BenchmarkInstance{(*std::move(result))};              \
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

[[nodiscard]] llvm::LogicalResult fail(const std::string_view source,
                                       const std::string_view pointer,
                                       const std::string_view message) {
  return ::mqt::emitError(std::string(source) + ":" + std::string(pointer) +
                              " " + std::string(message),
                          ::mqt::ErrorCategory::InvalidArgument);
}

template <class Factory>
[[nodiscard]] auto constructBenchmark(const std::string_view source,
                                      Factory&& factory) {
  ::mqt::ScopedDiagnosticHandler const context(
      [&](const ::mqt::Diagnostic& diagnostic) {
        auto located = diagnostic;
        located.message =
            std::string(source) + ":$/parameters " + located.message;
        ::mqt::emitDiagnostic(located);
        return llvm::success();
      });
  return std::forward<Factory>(factory)();
}

[[nodiscard]] llvm::FailureOr<Json> parseJSON(const std::string_view text,
                                              const std::string_view source) {
  std::vector<std::unordered_set<std::string>> keysByDepth;
  auto duplicate = llvm::success();
  const auto rejectDuplicates = [&](const int depth,
                                    const Json::parse_event_t event,
                                    Json& parsed) {
    if (event == Json::parse_event_t::object_start) {
      const auto index = static_cast<size_t>(depth);
      if (keysByDepth.size() <= index) {
        keysByDepth.resize(index + 1U);
      }
      keysByDepth[index].clear();
    } else if (event == Json::parse_event_t::key) {
      const auto index = static_cast<size_t>(depth - 1);
      const auto& key = parsed.get_ref<const std::string&>();
      if (!keysByDepth[index].emplace(key).second &&
          llvm::succeeded(duplicate)) {
        duplicate = fail(source, "$", "contains duplicate key '" + key + "'");
      }
    }
    return true;
  };
  auto result = mqt::detail::parseJSON(text, source, rejectDuplicates);
  if (llvm::failed(duplicate)) {
    return llvm::failure();
  }
  return result;
}

llvm::LogicalResult requireObject(const Json& value,
                                  const std::string_view source,
                                  const std::string_view pointer) {
  if (!value.is_object()) {
    return fail(source, pointer, "must be an object");
  }
  return llvm::success();
}

llvm::LogicalResult rejectUnknownKeys(
    const Json& value, const std::initializer_list<std::string_view> known,
    const std::string_view source, const std::string_view pointer) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(known, key) == known.end()) {
      return fail(source, pointer, "contains unknown key '" + key + "'");
    }
  }
  return llvm::success();
}

[[nodiscard]] llvm::FailureOr<const Json*>
required(const Json& value, const char* const key,
         const std::string_view source, const std::string_view pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    return fail(source, std::string(pointer) + "/" + key, "is required");
  }
  return &*found;
}

[[nodiscard]] llvm::FailureOr<uint64_t>
unsignedInteger(const Json& value, const std::string_view source,
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

[[nodiscard]] llvm::FailureOr<size_t>
sizeValue(const Json& value, const std::string_view source,
          const std::string_view pointer) {
  auto result = unsignedInteger(value, source, pointer);
  if (llvm::failed(result)) {
    return llvm::failure();
  }
  const auto parsed = (*result);
  if (parsed > std::numeric_limits<size_t>::max()) {
    return fail(source, pointer, "must fit size_t");
  }
  return static_cast<size_t>(parsed);
}

[[nodiscard]] llvm::FailureOr<std::string>
stringValue(const Json& value, const std::string_view source,
            const std::string_view pointer) {
  if (!value.is_string()) {
    return fail(source, pointer, "must be a string");
  }
  return value.get<std::string>();
}

[[nodiscard]] llvm::FailureOr<double>
numberValue(const Json& value, const std::string_view source,
            const std::string_view pointer) {
  if (!value.is_number()) {
    return fail(source, pointer, "must be a number");
  }
  return value.get<double>();
}

llvm::LogicalResult requireSchemaVersion(const Json& root,
                                         const std::string_view source) {
  auto schemaVersionField = required(root, "schema_version", source, "$");
  if (llvm::failed(schemaVersionField)) {
    return llvm::failure();
  }
  auto schemaVersionValue =
      unsignedInteger(*(*schemaVersionField), source, "$/schema_version");
  if (llvm::failed(schemaVersionValue)) {
    return llvm::failure();
  }
  const auto version = (*schemaVersionValue);
  if (version != SCHEMA_VERSION) {
    return fail(source, "$/schema_version", "must be 1");
  }
  return llvm::success();
}

[[nodiscard]] llvm::FailureOr<const RegistryEntry*>
requireBenchmarkEntry(const Json& root, const std::string_view source) {
  auto benchmarkField = required(root, "benchmark", source, "$");
  if (llvm::failed(benchmarkField)) {
    return llvm::failure();
  }
  auto benchmarkValue = stringValue(*(*benchmarkField), source, "$/benchmark");
  if (llvm::failed(benchmarkValue)) {
    return llvm::failure();
  }
  const auto& benchmark = (*benchmarkValue);
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry;
  }
  return fail(source, "$/benchmark",
              "selects unsupported benchmark '" + benchmark + "'");
}

[[nodiscard]] llvm::FailureOr<Json> envelope(const std::string_view text,
                                             const std::string_view source,
                                             const bool manifest) {
  auto parsed = parseJSON(text, source);
  if (llvm::failed(parsed)) {
    return llvm::failure();
  }
  auto& root = (*parsed);
  if (llvm::failed(requireObject(root, source, "$"))) {
    return llvm::failure();
  }
  auto const keyError =
      manifest
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
          : rejectUnknownKeys(root,
                              {"schema_version", "benchmark", "parameters"},
                              source, "$");
  if (llvm::failed(keyError)) {
    return llvm::failure();
  }
  if (llvm::failed(requireSchemaVersion(root, source))) {
    return llvm::failure();
  }
  auto entry = requireBenchmarkEntry(root, source);
  if (llvm::failed(entry)) {
    return llvm::failure();
  }
  auto parameters = required(root, "parameters", source, "$");
  if (llvm::failed(parameters)) {
    return llvm::failure();
  }
  if (llvm::failed(requireObject(*(*parameters), source, "$/parameters"))) {
    return llvm::failure();
  }
  if (manifest) {
    auto definitionVersionField =
        required(root, "definition_version", source, "$");
    if (llvm::failed(definitionVersionField)) {
      return llvm::failure();
    }
    auto definitionVersionValue = unsignedInteger(
        *(*definitionVersionField), source, "$/definition_version");
    if (llvm::failed(definitionVersionValue)) {
      return llvm::failure();
    }
    const auto definition = (*definitionVersionValue);
    if (definition != (*entry)->definitionVersion) {
      return fail(source, "$/definition_version",
                  "must be " + std::to_string((*entry)->definitionVersion));
    }
    auto caseIdField = required(root, "case_id", source, "$");
    if (llvm::failed(caseIdField)) {
      return llvm::failure();
    }
    auto caseIdValue = stringValue(*(*caseIdField), source, "$/case_id");
    if (llvm::failed(caseIdValue)) {
      return llvm::failure();
    }
    const auto& caseId = (*caseIdValue);
    static_cast<void>(caseId);
    auto outputsField = required(root, "outputs", source, "$");
    if (llvm::failed(outputsField)) {
      return llvm::failure();
    }
    const auto& outputs = *(*outputsField);
    if (!outputs.is_array()) {
      return fail(source, "$/outputs", "must be an array");
    }
    auto referenceField = required(root, "reference", source, "$");
    if (llvm::failed(referenceField)) {
      return llvm::failure();
    }
    const auto& reference = *(*referenceField);
    if (llvm::failed(requireObject(reference, source, "$/reference"))) {
      return llvm::failure();
    }
  }
  return std::move(root);
}

llvm::LogicalResult requireBenchmark(const Json& root,
                                     const std::string_view expected,
                                     const std::string_view source) {
  const auto& actual = root["benchmark"].get_ref<const std::string&>();
  if (actual != expected) {
    return fail(source, "$/benchmark",
                "must be '" + std::string(expected) + "'");
  }
  return llvm::success();
}

[[nodiscard]] llvm::FailureOr<BV>
parseBVParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters, {"hidden_bitstring", "method"},
                                     source, "$/parameters"))) {
    return llvm::failure();
  }
  auto hiddenBitstringField =
      required(parameters, "hidden_bitstring", source, "$/parameters");
  if (llvm::failed(hiddenBitstringField)) {
    return llvm::failure();
  }
  auto hiddenBitstringValue = stringValue(*(*hiddenBitstringField), source,
                                          "$/parameters/hidden_bitstring");
  if (llvm::failed(hiddenBitstringValue)) {
    return llvm::failure();
  }
  BVOptions options{
      .hiddenBitstring = std::move(*hiddenBitstringValue),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    auto methodValue = stringValue(*method, source, "$/parameters/method");
    if (llvm::failed(methodValue)) {
      return llvm::failure();
    }
    const auto& value = (*methodValue);
    if (value == "static") {
      options.method = BVMethod::Static;
    } else if (value == "dynamic") {
      options.method = BVMethod::Dynamic;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'static' or 'dynamic'");
    }
  }
  return constructBenchmark(source,
                            [&] { return BV::create(std::move(options)); });
}

[[nodiscard]] llvm::FailureOr<WeakMeasurementGrover>
parseWeakMeasurementGroverParameters(const Json& parameters,
                                     const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(
          parameters, {"marked_bitstring", "measurement_strength"}, source,
          "$/parameters"))) {
    return llvm::failure();
  }
  auto field = required(parameters, "marked_bitstring", source, "$/parameters");
  if (llvm::failed(field)) {
    return llvm::failure();
  }
  auto marked = stringValue(**field, source, "$/parameters/marked_bitstring");
  if (llvm::failed(marked)) {
    return llvm::failure();
  }
  WeakMeasurementGroverOptions options{.markedBitstring = std::move(*marked)};
  if (const auto strength = parameters.find("measurement_strength");
      strength != parameters.end()) {
    const auto value =
        numberValue(*strength, source, "$/parameters/measurement_strength");
    if (llvm::failed(value)) {
      return llvm::failure();
    }
    options.measurementStrength = value;
  }
  return constructBenchmark(source, [&] {
    return WeakMeasurementGrover::create(std::move(options));
  });
}

[[nodiscard]] llvm::FailureOr<MagicStateDistillation>
parseMagicStateDistillationParameters(const Json& parameters,
                                      const std::string_view source) {
  if (llvm::failed(
          rejectUnknownKeys(parameters, {"levels"}, source, "$/parameters"))) {
    return llvm::failure();
  }
  MagicStateDistillationOptions options;
  if (const auto levels = parameters.find("levels");
      levels != parameters.end()) {
    auto value = sizeValue(*levels, source, "$/parameters/levels");
    if (llvm::failed(value)) {
      return llvm::failure();
    }
    options.levels = *value;
  }
  return constructBenchmark(
      source, [&] { return MagicStateDistillation::create(options); });
}

[[nodiscard]] llvm::FailureOr<ModularMultiplier>
parseModularMultiplierParameters(const Json& parameters,
                                 const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(
          parameters, {"multiplier", "modulus", "multiplicand", "control"},
          source, "$/parameters"))) {
    return llvm::failure();
  }
  auto control = std::string("1");
  if (const auto value = parameters.find("control");
      value != parameters.end()) {
    auto controlValue = stringValue(*value, source, "$/parameters/control");
    if (llvm::failed(controlValue)) {
      return llvm::failure();
    }
    control = std::move(*controlValue);
  }
  if (control.size() != 1U) {
    return fail(source, "$/parameters/control", "must be '0', '1', or '+'");
  }
  auto multiplicandField =
      required(parameters, "multiplicand", source, "$/parameters");
  if (llvm::failed(multiplicandField)) {
    return llvm::failure();
  }
  auto multiplicandValue =
      stringValue(*(*multiplicandField), source, "$/parameters/multiplicand");
  if (llvm::failed(multiplicandValue)) {
    return llvm::failure();
  }
  auto modulusField = required(parameters, "modulus", source, "$/parameters");
  if (llvm::failed(modulusField)) {
    return llvm::failure();
  }
  auto modulusValue =
      stringValue(*(*modulusField), source, "$/parameters/modulus");
  if (llvm::failed(modulusValue)) {
    return llvm::failure();
  }
  auto multiplierField =
      required(parameters, "multiplier", source, "$/parameters");
  if (llvm::failed(multiplierField)) {
    return llvm::failure();
  }
  auto multiplierValue =
      stringValue(*(*multiplierField), source, "$/parameters/multiplier");
  if (llvm::failed(multiplierValue)) {
    return llvm::failure();
  }
  return constructBenchmark(source, [&] {
    return ModularMultiplier::create({
        .multiplier = std::move(*multiplierValue),
        .modulus = std::move(*modulusValue),
        .multiplicand = std::move(*multiplicandValue),
        .control = control.front(),
    });
  });
}

[[nodiscard]] llvm::FailureOr<GHZ>
parseGHZParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters,
                                     {"qubits", "topology", "basis"}, source,
                                     "$/parameters"))) {
    return llvm::failure();
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (llvm::failed(qubitsField)) {
    return llvm::failure();
  }
  auto qubitsValue = sizeValue(*(*qubitsField), source, "$/parameters/qubits");
  if (llvm::failed(qubitsValue)) {
    return llvm::failure();
  }
  GHZOptions options{
      .qubits = (*qubitsValue),
  };
  if (const auto topology = parameters.find("topology");
      topology != parameters.end()) {
    auto topologyValue =
        stringValue(*topology, source, "$/parameters/topology");
    if (llvm::failed(topologyValue)) {
      return llvm::failure();
    }
    const auto& value = (*topologyValue);
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
    if (llvm::failed(basisValue)) {
      return llvm::failure();
    }
    const auto& value = (*basisValue);
    if (value == "z") {
      options.basis = GHZBasis::Z;
    } else if (value == "x") {
      options.basis = GHZBasis::X;
    } else {
      return fail(source, "$/parameters/basis", "must be 'z' or 'x'");
    }
  }
  return constructBenchmark(source, [&] { return GHZ::create(options); });
}

[[nodiscard]] llvm::FailureOr<Grover>
parseGroverParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters,
                                     {"marked_bitstring", "iterations"}, source,
                                     "$/parameters"))) {
    return llvm::failure();
  }
  auto markedBitstringField =
      required(parameters, "marked_bitstring", source, "$/parameters");
  if (llvm::failed(markedBitstringField)) {
    return llvm::failure();
  }
  auto markedBitstringValue = stringValue(*(*markedBitstringField), source,
                                          "$/parameters/marked_bitstring");
  if (llvm::failed(markedBitstringValue)) {
    return llvm::failure();
  }
  GroverOptions options{
      .markedBitstring = std::move(*markedBitstringValue),
  };
  if (const auto iterations = parameters.find("iterations");
      iterations != parameters.end()) {
    auto iterationsValue =
        sizeValue(*iterations, source, "$/parameters/iterations");
    if (llvm::failed(iterationsValue)) {
      return llvm::failure();
    }
    options.iterations.emplace(*iterationsValue);
  }
  return constructBenchmark(source,
                            [&] { return Grover::create(std::move(options)); });
}

[[nodiscard]] llvm::FailureOr<Multiplexer>
parseMultiplexerParameters(const Json& parameters,
                           const std::string_view source) {
  if (llvm::failed(
          rejectUnknownKeys(parameters, {"qubits"}, source, "$/parameters"))) {
    return llvm::failure();
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (llvm::failed(qubitsField)) {
    return llvm::failure();
  }
  auto qubitsValue = sizeValue(*(*qubitsField), source, "$/parameters/qubits");
  if (llvm::failed(qubitsValue)) {
    return llvm::failure();
  }
  return constructBenchmark(source, [&] {
    return Multiplexer::create({
        .qubits = (*qubitsValue),
    });
  });
}

[[nodiscard]] llvm::FailureOr<QFT>
parseQFTParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters,
                                     {"qubits", "period_exponent", "method"},
                                     source, "$/parameters"))) {
    return llvm::failure();
  }
  auto periodExponentField =
      required(parameters, "period_exponent", source, "$/parameters");
  if (llvm::failed(periodExponentField)) {
    return llvm::failure();
  }
  auto periodExponentValue = sizeValue(*(*periodExponentField), source,
                                       "$/parameters/period_exponent");
  if (llvm::failed(periodExponentValue)) {
    return llvm::failure();
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (llvm::failed(qubitsField)) {
    return llvm::failure();
  }
  auto qubitsValue = sizeValue(*(*qubitsField), source, "$/parameters/qubits");
  if (llvm::failed(qubitsValue)) {
    return llvm::failure();
  }
  QFTOptions options{
      .qubits = (*qubitsValue),
      .periodExponent = (*periodExponentValue),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    auto methodValue = stringValue(*method, source, "$/parameters/method");
    if (llvm::failed(methodValue)) {
      return llvm::failure();
    }
    const auto& value = (*methodValue);
    if (value == "standard") {
      options.method = QFTMethod::Standard;
    } else if (value == "semiclassical") {
      options.method = QFTMethod::Semiclassical;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'standard' or 'semiclassical'");
    }
  }
  return constructBenchmark(source, [&] { return QFT::create(options); });
}

[[nodiscard]] llvm::FailureOr<QFTAdder>
parseQFTAdderParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(
          parameters, {"addend", "accumulator", "method", "overflow"}, source,
          "$/parameters"))) {
    return llvm::failure();
  }
  auto accumulatorField =
      required(parameters, "accumulator", source, "$/parameters");
  if (llvm::failed(accumulatorField)) {
    return llvm::failure();
  }
  auto accumulatorValue =
      stringValue(*(*accumulatorField), source, "$/parameters/accumulator");
  if (llvm::failed(accumulatorValue)) {
    return llvm::failure();
  }
  auto addendField = required(parameters, "addend", source, "$/parameters");
  if (llvm::failed(addendField)) {
    return llvm::failure();
  }
  auto addendValue =
      stringValue(*(*addendField), source, "$/parameters/addend");
  if (llvm::failed(addendValue)) {
    return llvm::failure();
  }
  QFTAdderOptions options{
      .addend = std::move(*addendValue),
      .accumulator = std::move(*accumulatorValue),
  };
  if (const auto it = parameters.find("method"); it != parameters.end()) {
    auto methodValue = stringValue(*it, source, "$/parameters/method");
    if (llvm::failed(methodValue)) {
      return llvm::failure();
    }
    const auto& value = (*methodValue);
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
    if (llvm::failed(overflowValue)) {
      return llvm::failure();
    }
    const auto& value = (*overflowValue);
    if (value == "wrap") {
      options.overflow = QFTAdderOverflow::Wrap;
    } else if (value == "carry") {
      options.overflow = QFTAdderOverflow::Carry;
    } else {
      return fail(source, "$/parameters/overflow", "must be 'wrap' or 'carry'");
    }
  }
  return constructBenchmark(
      source, [&] { return QFTAdder::create(std::move(options)); });
}

[[nodiscard]] llvm::FailureOr<QPE>
parseQPEParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters,
                                     {"precision", "phase", "method"}, source,
                                     "$/parameters"))) {
    return llvm::failure();
  }
  auto precisionField =
      required(parameters, "precision", source, "$/parameters");
  if (llvm::failed(precisionField)) {
    return llvm::failure();
  }
  auto precisionValue =
      sizeValue(*(*precisionField), source, "$/parameters/precision");
  if (llvm::failed(precisionValue)) {
    return llvm::failure();
  }
  const auto precision = (*precisionValue);
  auto phaseField = required(parameters, "phase", source, "$/parameters");
  if (llvm::failed(phaseField)) {
    return llvm::failure();
  }
  const auto& phase = *(*phaseField);
  if (llvm::failed(requireObject(phase, source, "$/parameters/phase"))) {
    return llvm::failure();
  }
  if (llvm::failed(rejectUnknownKeys(phase, {"numerator", "denominator"},
                                     source, "$/parameters/phase"))) {
    return llvm::failure();
  }
  auto numeratorField =
      required(phase, "numerator", source, "$/parameters/phase");
  if (llvm::failed(numeratorField)) {
    return llvm::failure();
  }
  auto numeratorValue = unsignedInteger(*(*numeratorField), source,
                                        "$/parameters/phase/numerator");
  if (llvm::failed(numeratorValue)) {
    return llvm::failure();
  }
  const auto numerator = (*numeratorValue);
  auto denominatorField =
      required(phase, "denominator", source, "$/parameters/phase");
  if (llvm::failed(denominatorField)) {
    return llvm::failure();
  }
  auto denominatorValue = unsignedInteger(*(*denominatorField), source,
                                          "$/parameters/phase/denominator");
  if (llvm::failed(denominatorValue)) {
    return llvm::failure();
  }
  const auto denominator = (*denominatorValue);
  auto method = QPEMethod::Standard;
  if (const auto value = parameters.find("method"); value != parameters.end()) {
    auto methodValue = stringValue(*value, source, "$/parameters/method");
    if (llvm::failed(methodValue)) {
      return llvm::failure();
    }
    const auto& name = (*methodValue);
    if (name == "standard") {
      method = QPEMethod::Standard;
    } else if (name == "iterative") {
      method = QPEMethod::Iterative;
    } else {
      return fail(source, "$/parameters/method",
                  "must be 'standard' or 'iterative'");
    }
  }
  auto phaseValue = constructBenchmark(
      source, [&] { return Phase::create(numerator, denominator); });
  if (llvm::failed(phaseValue)) {
    return llvm::failure();
  }
  return constructBenchmark(source, [&] {
    return QPE::create({
        .precision = precision,
        .phase = (*phaseValue),
        .method = method,
    });
  });
}

[[nodiscard]] llvm::FailureOr<RepeatUntilSuccess>
parseRepeatUntilSuccessParameters(const Json& parameters,
                                  const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters, {"data_qubits"}, source,
                                     "$/parameters"))) {
    return llvm::failure();
  }
  RepeatUntilSuccessOptions options;
  if (const auto width = parameters.find("data_qubits");
      width != parameters.end()) {
    auto dataQubitsValue =
        sizeValue(*width, source, "$/parameters/data_qubits");
    if (llvm::failed(dataQubitsValue)) {
      return llvm::failure();
    }
    options.dataQubits = (*dataQubitsValue);
  }
  return constructBenchmark(
      source, [&] { return RepeatUntilSuccess::create(options); });
}

[[nodiscard]] llvm::FailureOr<Teleportation>
parseTeleportationParameters(const Json& parameters,
                             const std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters, {}, source, "$/parameters"))) {
    return llvm::failure();
  }
  return Teleportation{};
}

[[nodiscard]] llvm::FailureOr<WState>
parseWStateParameters(const Json& parameters, const std::string_view source) {
  if (llvm::failed(
          rejectUnknownKeys(parameters, {"qubits"}, source, "$/parameters"))) {
    return llvm::failure();
  }
  auto field = required(parameters, "qubits", source, "$/parameters");
  if (llvm::failed(field)) {
    return llvm::failure();
  }
  auto qubits = sizeValue(**field, source, "$/parameters/qubits");
  if (llvm::failed(qubits)) {
    return llvm::failure();
  }
  return constructBenchmark(
      source, [&] { return WState::create({.qubits = *qubits}); });
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

[[nodiscard]] llvm::FailureOr<Shor>
parseShorParameters(const Json& parameters, std::string_view source) {
  if (llvm::failed(rejectUnknownKeys(parameters, {"number", "base"}, source,
                                     "$/parameters"))) {
    return llvm::failure();
  }
  auto field = required(parameters, "number", source, "$/parameters");
  if (llvm::failed(field)) {
    return llvm::failure();
  }
  auto number = unsignedInteger(**field, source, "$/parameters/number");
  if (llvm::failed(number)) {
    return llvm::failure();
  }
  ShorOptions options{.number = *number};
  if (const auto base = parameters.find("base"); base != parameters.end()) {
    auto value = unsignedInteger(*base, source, "$/parameters/base");
    if (llvm::failed(value)) {
      return llvm::failure();
    }
    options.base = *value;
  }
  return constructBenchmark(source, [&] { return Shor::create(options); });
}

[[nodiscard]] Json parametersJSON(const Shor& benchmark) {
  const auto& options = benchmark.options();
  return {{"number", options.number}, {"base", options.base}};
}

[[nodiscard]] Json parametersJSON(const BV& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"hidden_bitstring", options.hiddenBitstring},
      {"method", methodName(options.method)},
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

[[nodiscard]] Json parametersJSON(const WeakMeasurementGrover& benchmark) {
  const auto& options = benchmark.options();
  return {
      {"marked_bitstring", options.markedBitstring},
      {"measurement_strength", *options.measurementStrength},
  };
}

[[nodiscard]] Json parametersJSON(const MagicStateDistillation& benchmark) {
  return {{"levels", benchmark.options().levels}};
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

[[nodiscard]] Json parametersJSON(const WState& benchmark) {
  return {{"qubits", benchmark.options().qubits}};
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

[[nodiscard]] Json referenceJSON(const Shor& benchmark) {
  return {
      {"kind", "verification"},
      {"model", "shor_factors"},
      {"outcome_order", "big_endian"},
      {"output", benchmark.output().name},
      {"version", 1},
  };
}

[[nodiscard]] Json referenceJSON(const BV& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "bernstein_vazirani",
                               benchmark.options().hiddenBitstring);
}

[[nodiscard]] Json referenceJSON(const GHZ& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "ghz");
}

[[nodiscard]] Json referenceJSON(const Grover& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "grover_single_marked",
                               benchmark.options().markedBitstring);
}

[[nodiscard]] Json referenceJSON(const WeakMeasurementGrover& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "grover_weak_measurement",
                               benchmark.options().markedBitstring);
}

[[nodiscard]] Json referenceJSON(const MagicStateDistillation& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "magic_state_distillation",
                               "00");
}

[[nodiscard]] Json referenceJSON(const ModularMultiplier& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "modular_multiplier",
                               benchmark.expectedResult());
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

[[nodiscard]] Json referenceJSON(const WState& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "w_state");
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

[[nodiscard]] std::string semanticCaseId(const Json& semantic) {
  auto input = std::string(CASE_DOMAIN);
  input.push_back('\0');
  input += semantic.dump();
  return "sha256-" +
         llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(input)),
                     true);
}

template <class Benchmark>
[[nodiscard]] Json manifestJSON(const Benchmark& benchmark) {
  auto semantic = semanticJSON(benchmark);
  semantic["case_id"] = semanticCaseId(semantic);
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

template <class Benchmark>
[[nodiscard]] llvm::LogicalResult
requireManifest(const Json& root, const Benchmark& benchmark,
                const std::string_view source) {
  if (root.dump() != manifestJSON(benchmark).dump()) {
    return fail(source, "$",
                "does not match its resolved benchmark instance and case ID");
  }
  return llvm::success();
}

template <class Benchmark, class ParseParameters>
[[nodiscard]] llvm::FailureOr<Benchmark>
parseBenchmark(const std::string_view text, const std::string_view source,
               const ParseParameters& parseParameters, const bool manifest) {
  auto parsed = envelope(text, source, manifest);
  if (llvm::failed(parsed)) {
    return llvm::failure();
  }
  const auto& root = (*parsed);
  if (llvm::failed(
          requireBenchmark(root, BenchmarkMetadata<Benchmark>::id, source))) {
    return llvm::failure();
  }
  auto benchmark = parseParameters(root["parameters"], source);
  if (llvm::failed(benchmark)) {
    return llvm::failure();
  }
  if (manifest && llvm::failed(requireManifest(root, *benchmark, source))) {
    return llvm::failure();
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

[[nodiscard]] Json weakMeasurementGroverInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<WeakMeasurementGrover>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "marked_bitstring",
                  {
                      {"maxLength", WeakMeasurementGroverOptions::MAX_QUBITS},
                      {"minLength", 2},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
              {
                  "measurement_strength",
                  {
                      {"exclusiveMinimum", 0},
                      {"maximum", 0.5},
                      {"type", "number"},
                  },
              },
          },
      },
      {"required", {"marked_bitstring"}},
      {"type", "object"},
  });
}

[[nodiscard]] Json magicStateDistillationInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<MagicStateDistillation>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "levels",
                  {
                      {"default", 1},
                      {"minimum", 1},
                      {"maximum", std::numeric_limits<int64_t>::max() / 5},
                      {"type", "integer"},
                  },
              },
          },
      },
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

[[nodiscard]] Json shorInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Shor>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "number",
                  {
                      {"type", "integer"},
                      {"minimum", 3},
                      {"maximum", ShorOptions::MAX_NUMBER},
                      {"not", {{"multipleOf", 2}}},
                  },
              },
              {
                  "base",
                  {
                      {"type", "integer"},
                      {"minimum", 2},
                      {"maximum", ShorOptions::MAX_NUMBER - 1},
                      {"default", 2},
                  },
              },
          },
      },
      {"required", {"number"}},
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

[[nodiscard]] Json wStateInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<WState>({
      {"additionalProperties", false},
      {
          "properties",
          {
              {
                  "qubits",
                  {
                      {"maximum", std::numeric_limits<int64_t>::max()},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", {"qubits"}},
      {"type", "object"},
  });
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

llvm::FailureOr<std::string>
benchmarkIdFromInstanceSpecificationJSON(const std::string_view json,
                                         const std::string_view source) {
  auto result = envelope(json, source, false);
  if (llvm::failed(result)) {
    return llvm::failure();
  }
  return (*result)["benchmark"].get<std::string>();
}

llvm::FailureOr<std::string>
benchmarkIdFromManifestJSON(const std::string_view json,
                            const std::string_view source) {
  auto result = envelope(json, source, true);
  if (llvm::failed(result)) {
    return llvm::failure();
  }
  return (*result)["benchmark"].get<std::string>();
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

llvm::FailureOr<std::string>
describeBenchmarkJSON(const std::string_view benchmark) {
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry->instanceSpecificationSchema().dump();
  }
  return ::mqt::emitError("unsupported benchmark '" + std::string(benchmark) +
                              "'",
                          ::mqt::ErrorCategory::InvalidArgument);
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  llvm::FailureOr<TYPE> STEM##FromInstanceSpecificationJSON(                   \
      const std::string_view json, const std::string_view source) {            \
    return parseBenchmark<TYPE>(json, source, parse##TYPE##Parameters, false); \
  }                                                                            \
  std::string toInstanceSpecificationJSON(const TYPE& benchmark) {             \
    return instanceSpecificationJSON(benchmark).dump();                        \
  }                                                                            \
  llvm::FailureOr<TYPE> STEM##FromManifestJSON(                                \
      const std::string_view json, const std::string_view source) {            \
    return parseBenchmark<TYPE>(json, source, parse##TYPE##Parameters, true);  \
  }                                                                            \
  std::string toManifestJSON(const TYPE& benchmark) {                          \
    return manifestJSON(benchmark).dump();                                     \
  }                                                                            \
  std::string caseId(const TYPE& benchmark) {                                  \
    return semanticCaseId(semanticJSON(benchmark));                            \
  }
#include "bench/BenchmarkFamilies.inc"

llvm::FailureOr<Counts> countsFromJSON(const std::string_view json,
                                       const std::string_view source) {
  auto parsed = parseJSON(json, source);
  if (llvm::failed(parsed)) {
    return llvm::failure();
  }
  const auto& root = (*parsed);
  if (llvm::failed(requireObject(root, source, "$"))) {
    return llvm::failure();
  }
  if (llvm::failed(
          rejectUnknownKeys(root, {"schema_version", "counts"}, source, "$"))) {
    return llvm::failure();
  }
  if (llvm::failed(requireSchemaVersion(root, source))) {
    return llvm::failure();
  }
  auto field = required(root, "counts", source, "$");
  if (llvm::failed(field)) {
    return llvm::failure();
  }
  const auto& values = *(*field);
  if (llvm::failed(requireObject(values, source, "$/counts"))) {
    return llvm::failure();
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
    if (llvm::failed(countResult)) {
      return llvm::failure();
    }
    const auto count = (*countResult);
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

llvm::FailureOr<std::string> evaluateJSON(const std::string_view manifest,
                                          const std::string_view counts,
                                          const std::string_view manifestSource,
                                          const std::string_view countsSource) {
  auto parsed = envelope(manifest, manifestSource, true);
  if (llvm::failed(parsed)) {
    return llvm::failure();
  }
  auto parsedCounts = countsFromJSON(counts, countsSource);
  if (llvm::failed(parsedCounts)) {
    return llvm::failure();
  }
  const auto& root = *parsed;
  const auto& id = root["benchmark"].get_ref<const std::string&>();
  auto instance = findBenchmark(id)->parse(root["parameters"], manifestSource);
  if (llvm::failed(instance)) {
    return llvm::failure();
  }
  return std::visit(
      [&](const auto& benchmark) -> llvm::FailureOr<std::string> {
        if (llvm::failed(requireManifest(root, benchmark, manifestSource))) {
          return llvm::failure();
        }
        auto evaluation = benchmark.evaluate(*parsedCounts);
        if (llvm::failed(evaluation)) {
          return llvm::failure();
        }
        // Evaluation validates the total before this sum.
        const auto shots =
            std::accumulate(parsedCounts->begin(), parsedCounts->end(),
                            size_t{0}, [](const size_t sum, const auto& item) {
                              return sum + item.second;
                            });
        return evaluationToJSON(root["case_id"].get_ref<const std::string&>(),
                                shots, *evaluation);
      },
      *instance);
}

llvm::FailureOr<std::string>
evaluationToJSON(const std::string_view caseIdValue, const size_t shots,
                 const Evaluation& evaluation) {
  if (!validCaseId(caseIdValue)) {
    return ::mqt::emitError("case ID must be a full lowercase SHA-256 ID",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  if (shots == 0) {
    return ::mqt::emitError("evaluation requires at least one shot",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  const auto validMetric = [](const double value) {
    return std::isfinite(value) && value >= 0. && value <= 1.;
  };
  if (!validMetric(evaluation.totalVariationDistance) ||
      !validMetric(evaluation.squaredHellingerFidelity) ||
      (evaluation.successProbability &&
       !validMetric(*evaluation.successProbability))) {
    return ::mqt::emitError("evaluation metrics must be finite and in [0, 1]",
                            ::mqt::ErrorCategory::InvalidArgument);
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

llvm::FailureOr<std::string>
evaluationToJSON(std::string_view caseIdValue, size_t shots,
                 const ShorEvaluation& evaluation) {
  if (!validCaseId(caseIdValue) || shots == 0) {
    return ::mqt::emitError(
        "evaluation requires a SHA-256 case ID and at least one shot",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (!std::isfinite(evaluation.successProbability) ||
      evaluation.successProbability < 0. ||
      evaluation.successProbability > 1. ||
      evaluation.factors.has_value() != (evaluation.successProbability > 0.)) {
    return ::mqt::emitError("factor verification requires a success "
                            "fraction in [0, 1] and factors on success",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  Json factors = nullptr;
  if (evaluation.factors) {
    const auto [first, second] = *evaluation.factors;
    if (first < 2 || first > second ||
        second > ShorOptions::MAX_NUMBER / first) {
      return ::mqt::emitError("factors must form a sorted nontrivial pair "
                              "within the supported range",
                              ::mqt::ErrorCategory::InvalidArgument);
    }
    factors = Json::array({first, second});
  }
  return Json{
      {"case_id", std::string(caseIdValue)},
      {"factors", factors},
      {"metrics", {{"success_probability", evaluation.successProbability}}},
      {"schema_version", SCHEMA_VERSION},
      {"shots", shots},
  }
      .dump();
}

llvm::FailureOr<ParsedBenchmark>
parseInstanceSpecificationJSON(const std::string_view json,
                               const std::string_view source) {
  auto parsed = envelope(json, source, false);
  if (llvm::failed(parsed)) {
    return llvm::failure();
  }
  const auto& root = *parsed;
  const auto& id = root["benchmark"].get_ref<const std::string&>();
  auto instance = findBenchmark(id)->parse(root["parameters"], source);
  if (llvm::failed(instance)) {
    return llvm::failure();
  }
  return std::visit(
      [&](auto&& benchmark) {
        const Json manifest = manifestJSON(benchmark);
        return ParsedBenchmark{
            .instance = std::forward<decltype(benchmark)>(benchmark),
            .benchmarkId = id,
            .caseId = manifest["case_id"].get<std::string>(),
            .manifestJSON = manifest.dump(),
        };
      },
      (*std::move(instance)));
}

} // namespace mqt::bench
