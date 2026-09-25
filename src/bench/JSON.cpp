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
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Shor.hpp"
#include "bench/Teleportation.hpp"
#include "bench/WState.hpp"

#include "JSON.hpp"
#include "support/Diagnostics.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringExtras.h"
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
  [[nodiscard]] mlir::FailureOr<std::string> evaluate##TYPE(                   \
      std::string_view manifest, std::string_view source,                      \
      const Counts& counts);
#include "bench/BenchmarkFamilies.inc"

using InstanceSpecificationSchemaFunction = Json (*)();
using EvaluationFunction = mlir::FailureOr<std::string> (*)(std::string_view,
                                                            std::string_view,
                                                            const Counts&);

struct RegistryEntry {
  std::string_view id;
  uint64_t definitionVersion;
  InstanceSpecificationSchemaFunction instanceSpecificationSchema;
  EvaluationFunction evaluate;
  mlir::FailureOr<BenchmarkInstance> (*parse)(std::string_view,
                                              std::string_view);
};
constexpr std::array REGISTRY{
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  RegistryEntry{                                                               \
      .id = (ID),                                                              \
      .definitionVersion = (DEFINITION_VERSION),                               \
      .instanceSpecificationSchema = STEM##InstanceSpecificationSchema,        \
      .evaluate = evaluate##TYPE,                                              \
      .parse =                                                                 \
          +[](std::string_view json,                                           \
              std::string_view source) -> mlir::FailureOr<BenchmarkInstance> { \
        auto result = STEM##FromInstanceSpecificationJSON(json, source);       \
        if (mlir::failed(result)) {                                            \
          return mlir::failure();                                              \
        }                                                                      \
        return BenchmarkInstance{(*std::move(result))};                        \
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

[[nodiscard]] mlir::LogicalResult fail(const std::string_view source,
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
        return mlir::success();
      });
  return std::forward<Factory>(factory)();
}

[[nodiscard]] mlir::FailureOr<Json> parseJSON(const std::string_view text,
                                              const std::string_view source) {
  std::vector<std::unordered_set<std::string>> keysByDepth;
  auto duplicate = mlir::success();
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
          mlir::succeeded(duplicate)) {
        duplicate = fail(source, "$", "contains duplicate key '" + key + "'");
      }
    }
    return true;
  };
  auto result = mqt::detail::parseJSON(text, source, rejectDuplicates);
  if (mlir::failed(duplicate)) {
    return mlir::failure();
  }
  return result;
}

mlir::LogicalResult requireObject(const Json& value,
                                  const std::string_view source,
                                  const std::string_view pointer) {
  if (!value.is_object()) {
    return fail(source, pointer, "must be an object");
  }
  return mlir::success();
}

mlir::LogicalResult rejectUnknownKeys(
    const Json& value, const std::initializer_list<std::string_view> known,
    const std::string_view source, const std::string_view pointer) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(known, key) == known.end()) {
      return fail(source, pointer, "contains unknown key '" + key + "'");
    }
  }
  return mlir::success();
}

[[nodiscard]] mlir::FailureOr<const Json*>
required(const Json& value, const char* const key,
         const std::string_view source, const std::string_view pointer) {
  const auto found = value.find(key);
  if (found == value.end()) {
    return fail(source, std::string(pointer) + "/" + key, "is required");
  }
  return &*found;
}

[[nodiscard]] mlir::FailureOr<uint64_t>
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

[[nodiscard]] mlir::FailureOr<size_t>
sizeValue(const Json& value, const std::string_view source,
          const std::string_view pointer) {
  auto result = unsignedInteger(value, source, pointer);
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  const auto parsed = (*result);
  if (parsed > std::numeric_limits<size_t>::max()) {
    return fail(source, pointer, "must fit size_t");
  }
  return static_cast<size_t>(parsed);
}

[[nodiscard]] mlir::FailureOr<std::string>
stringValue(const Json& value, const std::string_view source,
            const std::string_view pointer) {
  if (!value.is_string()) {
    return fail(source, pointer, "must be a string");
  }
  return value.get<std::string>();
}

mlir::LogicalResult requireSchemaVersion(const Json& root,
                                         const std::string_view source) {
  auto schemaVersionField = required(root, "schema_version", source, "$");
  if (mlir::failed(schemaVersionField)) {
    return mlir::failure();
  }
  auto schemaVersionValue =
      unsignedInteger(*(*schemaVersionField), source, "$/schema_version");
  if (mlir::failed(schemaVersionValue)) {
    return mlir::failure();
  }
  const auto version = (*schemaVersionValue);
  if (version != SCHEMA_VERSION) {
    return fail(source, "$/schema_version", "must be 1");
  }
  return mlir::success();
}

[[nodiscard]] mlir::FailureOr<const RegistryEntry*>
requireBenchmarkEntry(const Json& root, const std::string_view source) {
  auto benchmarkField = required(root, "benchmark", source, "$");
  if (mlir::failed(benchmarkField)) {
    return mlir::failure();
  }
  auto benchmarkValue = stringValue(*(*benchmarkField), source, "$/benchmark");
  if (mlir::failed(benchmarkValue)) {
    return mlir::failure();
  }
  const auto& benchmark = (*benchmarkValue);
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry;
  }
  return fail(source, "$/benchmark",
              "selects unsupported benchmark '" + benchmark + "'");
}

[[nodiscard]] mlir::FailureOr<Json> envelope(const std::string_view text,
                                             const std::string_view source,
                                             const bool manifest) {
  auto parsed = parseJSON(text, source);
  if (mlir::failed(parsed)) {
    return mlir::failure();
  }
  auto& root = (*parsed);
  if (mlir::failed(requireObject(root, source, "$"))) {
    return mlir::failure();
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
  if (mlir::failed(keyError)) {
    return mlir::failure();
  }
  if (mlir::failed(requireSchemaVersion(root, source))) {
    return mlir::failure();
  }
  auto entry = requireBenchmarkEntry(root, source);
  if (mlir::failed(entry)) {
    return mlir::failure();
  }
  auto parameters = required(root, "parameters", source, "$");
  if (mlir::failed(parameters)) {
    return mlir::failure();
  }
  if (mlir::failed(requireObject(*(*parameters), source, "$/parameters"))) {
    return mlir::failure();
  }
  if (manifest) {
    auto definitionVersionField =
        required(root, "definition_version", source, "$");
    if (mlir::failed(definitionVersionField)) {
      return mlir::failure();
    }
    auto definitionVersionValue = unsignedInteger(
        *(*definitionVersionField), source, "$/definition_version");
    if (mlir::failed(definitionVersionValue)) {
      return mlir::failure();
    }
    const auto definition = (*definitionVersionValue);
    if (definition != (*entry)->definitionVersion) {
      return fail(source, "$/definition_version",
                  "must be " + std::to_string((*entry)->definitionVersion));
    }
    auto caseIdField = required(root, "case_id", source, "$");
    if (mlir::failed(caseIdField)) {
      return mlir::failure();
    }
    auto caseIdValue = stringValue(*(*caseIdField), source, "$/case_id");
    if (mlir::failed(caseIdValue)) {
      return mlir::failure();
    }
    const auto& caseId = (*caseIdValue);
    static_cast<void>(caseId);
    auto outputsField = required(root, "outputs", source, "$");
    if (mlir::failed(outputsField)) {
      return mlir::failure();
    }
    const auto& outputs = *(*outputsField);
    if (!outputs.is_array()) {
      return fail(source, "$/outputs", "must be an array");
    }
    auto referenceField = required(root, "reference", source, "$");
    if (mlir::failed(referenceField)) {
      return mlir::failure();
    }
    const auto& reference = *(*referenceField);
    if (mlir::failed(requireObject(reference, source, "$/reference"))) {
      return mlir::failure();
    }
  }
  return std::move(root);
}

mlir::LogicalResult requireBenchmark(const Json& root,
                                     const std::string_view expected,
                                     const std::string_view source) {
  const auto& actual = root["benchmark"].get_ref<const std::string&>();
  if (actual != expected) {
    return fail(source, "$/benchmark",
                "must be '" + std::string(expected) + "'");
  }
  return mlir::success();
}

[[nodiscard]] mlir::FailureOr<BV>
parseBVParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters, {"hidden_bitstring", "method"},
                                     source, "$/parameters"))) {
    return mlir::failure();
  }
  auto hiddenBitstringField =
      required(parameters, "hidden_bitstring", source, "$/parameters");
  if (mlir::failed(hiddenBitstringField)) {
    return mlir::failure();
  }
  auto hiddenBitstringValue = stringValue(*(*hiddenBitstringField), source,
                                          "$/parameters/hidden_bitstring");
  if (mlir::failed(hiddenBitstringValue)) {
    return mlir::failure();
  }
  BVOptions options{
      .hiddenBitstring = (*hiddenBitstringValue),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    auto methodValue = stringValue(*method, source, "$/parameters/method");
    if (mlir::failed(methodValue)) {
      return mlir::failure();
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

[[nodiscard]] mlir::FailureOr<ModularMultiplier>
parseModularMultiplierParameters(const Json& parameters,
                                 const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(
          parameters, {"multiplier", "modulus", "multiplicand", "control"},
          source, "$/parameters"))) {
    return mlir::failure();
  }
  auto control = std::string("1");
  if (const auto value = parameters.find("control");
      value != parameters.end()) {
    auto controlValue = stringValue(*value, source, "$/parameters/control");
    if (mlir::failed(controlValue)) {
      return mlir::failure();
    }
    control = (*controlValue);
  }
  if (control.size() != 1U) {
    return fail(source, "$/parameters/control", "must be '0', '1', or '+'");
  }
  auto multiplicandField =
      required(parameters, "multiplicand", source, "$/parameters");
  if (mlir::failed(multiplicandField)) {
    return mlir::failure();
  }
  auto multiplicandValue =
      stringValue(*(*multiplicandField), source, "$/parameters/multiplicand");
  if (mlir::failed(multiplicandValue)) {
    return mlir::failure();
  }
  auto modulusField = required(parameters, "modulus", source, "$/parameters");
  if (mlir::failed(modulusField)) {
    return mlir::failure();
  }
  auto modulusValue =
      stringValue(*(*modulusField), source, "$/parameters/modulus");
  if (mlir::failed(modulusValue)) {
    return mlir::failure();
  }
  auto multiplierField =
      required(parameters, "multiplier", source, "$/parameters");
  if (mlir::failed(multiplierField)) {
    return mlir::failure();
  }
  auto multiplierValue =
      stringValue(*(*multiplierField), source, "$/parameters/multiplier");
  if (mlir::failed(multiplierValue)) {
    return mlir::failure();
  }
  return constructBenchmark(source, [&] {
    return ModularMultiplier::create({
        .multiplier = (*multiplierValue),
        .modulus = (*modulusValue),
        .multiplicand = (*multiplicandValue),
        .control = control.front(),
    });
  });
}

[[nodiscard]] mlir::FailureOr<GHZ>
parseGHZParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters,
                                     {"qubits", "topology", "basis"}, source,
                                     "$/parameters"))) {
    return mlir::failure();
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (mlir::failed(qubitsField)) {
    return mlir::failure();
  }
  auto qubitsValue = sizeValue(*(*qubitsField), source, "$/parameters/qubits");
  if (mlir::failed(qubitsValue)) {
    return mlir::failure();
  }
  GHZOptions options{
      .qubits = (*qubitsValue),
  };
  if (const auto topology = parameters.find("topology");
      topology != parameters.end()) {
    auto topologyValue =
        stringValue(*topology, source, "$/parameters/topology");
    if (mlir::failed(topologyValue)) {
      return mlir::failure();
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
    if (mlir::failed(basisValue)) {
      return mlir::failure();
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

[[nodiscard]] mlir::FailureOr<Grover>
parseGroverParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters,
                                     {"marked_bitstring", "iterations"}, source,
                                     "$/parameters"))) {
    return mlir::failure();
  }
  auto markedBitstringField =
      required(parameters, "marked_bitstring", source, "$/parameters");
  if (mlir::failed(markedBitstringField)) {
    return mlir::failure();
  }
  auto markedBitstringValue = stringValue(*(*markedBitstringField), source,
                                          "$/parameters/marked_bitstring");
  if (mlir::failed(markedBitstringValue)) {
    return mlir::failure();
  }
  GroverOptions options{
      .markedBitstring = (*markedBitstringValue),
  };
  if (const auto iterations = parameters.find("iterations");
      iterations != parameters.end()) {
    auto iterationsValue =
        sizeValue(*iterations, source, "$/parameters/iterations");
    if (mlir::failed(iterationsValue)) {
      return mlir::failure();
    }
    options.iterations.emplace(*iterationsValue);
  }
  return constructBenchmark(source,
                            [&] { return Grover::create(std::move(options)); });
}

[[nodiscard]] mlir::FailureOr<Multiplexer>
parseMultiplexerParameters(const Json& parameters,
                           const std::string_view source) {
  if (mlir::failed(
          rejectUnknownKeys(parameters, {"qubits"}, source, "$/parameters"))) {
    return mlir::failure();
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (mlir::failed(qubitsField)) {
    return mlir::failure();
  }
  auto qubitsValue = sizeValue(*(*qubitsField), source, "$/parameters/qubits");
  if (mlir::failed(qubitsValue)) {
    return mlir::failure();
  }
  return constructBenchmark(source, [&] {
    return Multiplexer::create({
        .qubits = (*qubitsValue),
    });
  });
}

[[nodiscard]] mlir::FailureOr<QFT>
parseQFTParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters,
                                     {"qubits", "period_exponent", "method"},
                                     source, "$/parameters"))) {
    return mlir::failure();
  }
  auto periodExponentField =
      required(parameters, "period_exponent", source, "$/parameters");
  if (mlir::failed(periodExponentField)) {
    return mlir::failure();
  }
  auto periodExponentValue = sizeValue(*(*periodExponentField), source,
                                       "$/parameters/period_exponent");
  if (mlir::failed(periodExponentValue)) {
    return mlir::failure();
  }
  auto qubitsField = required(parameters, "qubits", source, "$/parameters");
  if (mlir::failed(qubitsField)) {
    return mlir::failure();
  }
  auto qubitsValue = sizeValue(*(*qubitsField), source, "$/parameters/qubits");
  if (mlir::failed(qubitsValue)) {
    return mlir::failure();
  }
  QFTOptions options{
      .qubits = (*qubitsValue),
      .periodExponent = (*periodExponentValue),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    auto methodValue = stringValue(*method, source, "$/parameters/method");
    if (mlir::failed(methodValue)) {
      return mlir::failure();
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

[[nodiscard]] mlir::FailureOr<QFTAdder>
parseQFTAdderParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(
          parameters, {"addend", "accumulator", "method", "overflow"}, source,
          "$/parameters"))) {
    return mlir::failure();
  }
  auto accumulatorField =
      required(parameters, "accumulator", source, "$/parameters");
  if (mlir::failed(accumulatorField)) {
    return mlir::failure();
  }
  auto accumulatorValue =
      stringValue(*(*accumulatorField), source, "$/parameters/accumulator");
  if (mlir::failed(accumulatorValue)) {
    return mlir::failure();
  }
  auto addendField = required(parameters, "addend", source, "$/parameters");
  if (mlir::failed(addendField)) {
    return mlir::failure();
  }
  auto addendValue =
      stringValue(*(*addendField), source, "$/parameters/addend");
  if (mlir::failed(addendValue)) {
    return mlir::failure();
  }
  QFTAdderOptions options{
      .addend = (*addendValue),
      .accumulator = (*accumulatorValue),
  };
  if (const auto it = parameters.find("method"); it != parameters.end()) {
    auto methodValue = stringValue(*it, source, "$/parameters/method");
    if (mlir::failed(methodValue)) {
      return mlir::failure();
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
    if (mlir::failed(overflowValue)) {
      return mlir::failure();
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

[[nodiscard]] mlir::FailureOr<QPE>
parseQPEParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters,
                                     {"precision", "phase", "method"}, source,
                                     "$/parameters"))) {
    return mlir::failure();
  }
  auto precisionField =
      required(parameters, "precision", source, "$/parameters");
  if (mlir::failed(precisionField)) {
    return mlir::failure();
  }
  auto precisionValue =
      sizeValue(*(*precisionField), source, "$/parameters/precision");
  if (mlir::failed(precisionValue)) {
    return mlir::failure();
  }
  const auto precision = (*precisionValue);
  auto phaseField = required(parameters, "phase", source, "$/parameters");
  if (mlir::failed(phaseField)) {
    return mlir::failure();
  }
  const auto& phase = *(*phaseField);
  if (mlir::failed(requireObject(phase, source, "$/parameters/phase"))) {
    return mlir::failure();
  }
  if (mlir::failed(rejectUnknownKeys(phase, {"numerator", "denominator"},
                                     source, "$/parameters/phase"))) {
    return mlir::failure();
  }
  auto numeratorField =
      required(phase, "numerator", source, "$/parameters/phase");
  if (mlir::failed(numeratorField)) {
    return mlir::failure();
  }
  auto numeratorValue = unsignedInteger(*(*numeratorField), source,
                                        "$/parameters/phase/numerator");
  if (mlir::failed(numeratorValue)) {
    return mlir::failure();
  }
  const auto numerator = (*numeratorValue);
  auto denominatorField =
      required(phase, "denominator", source, "$/parameters/phase");
  if (mlir::failed(denominatorField)) {
    return mlir::failure();
  }
  auto denominatorValue = unsignedInteger(*(*denominatorField), source,
                                          "$/parameters/phase/denominator");
  if (mlir::failed(denominatorValue)) {
    return mlir::failure();
  }
  const auto denominator = (*denominatorValue);
  auto method = QPEMethod::Standard;
  if (const auto value = parameters.find("method"); value != parameters.end()) {
    auto methodValue = stringValue(*value, source, "$/parameters/method");
    if (mlir::failed(methodValue)) {
      return mlir::failure();
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
  if (mlir::failed(phaseValue)) {
    return mlir::failure();
  }
  return constructBenchmark(source, [&] {
    return QPE::create({
        .precision = precision,
        .phase = (*phaseValue),
        .method = method,
    });
  });
}

[[nodiscard]] mlir::FailureOr<RepeatUntilSuccess>
parseRepeatUntilSuccessParameters(const Json& parameters,
                                  const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters, {"data_qubits"}, source,
                                     "$/parameters"))) {
    return mlir::failure();
  }
  RepeatUntilSuccessOptions options;
  if (const auto width = parameters.find("data_qubits");
      width != parameters.end()) {
    auto dataQubitsValue =
        sizeValue(*width, source, "$/parameters/data_qubits");
    if (mlir::failed(dataQubitsValue)) {
      return mlir::failure();
    }
    options.dataQubits = (*dataQubitsValue);
  }
  return constructBenchmark(
      source, [&] { return RepeatUntilSuccess::create(options); });
}

[[nodiscard]] mlir::FailureOr<Teleportation>
parseTeleportationParameters(const Json& parameters,
                             const std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters, {}, source, "$/parameters"))) {
    return mlir::failure();
  }
  return Teleportation{};
}

[[nodiscard]] mlir::FailureOr<WState>
parseWStateParameters(const Json& parameters, const std::string_view source) {
  if (mlir::failed(
          rejectUnknownKeys(parameters, {"qubits"}, source, "$/parameters"))) {
    return mlir::failure();
  }
  auto field = required(parameters, "qubits", source, "$/parameters");
  if (mlir::failed(field)) {
    return mlir::failure();
  }
  auto qubits = sizeValue(**field, source, "$/parameters/qubits");
  if (mlir::failed(qubits)) {
    return mlir::failure();
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

[[nodiscard]] mlir::FailureOr<Shor>
parseShorParameters(const Json& parameters, std::string_view source) {
  if (mlir::failed(rejectUnknownKeys(parameters, {"number", "base"}, source,
                                     "$/parameters"))) {
    return mlir::failure();
  }
  auto field = required(parameters, "number", source, "$/parameters");
  if (mlir::failed(field)) {
    return mlir::failure();
  }
  auto number = unsignedInteger(**field, source, "$/parameters/number");
  if (mlir::failed(number)) {
    return mlir::failure();
  }
  ShorOptions options{.number = *number};
  if (const auto base = parameters.find("base"); base != parameters.end()) {
    auto value = unsignedInteger(*base, source, "$/parameters/base");
    if (mlir::failed(value)) {
      return mlir::failure();
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

template <class Benchmark, class ParseParameters>
[[nodiscard]] mlir::FailureOr<Benchmark>
parseBenchmark(const std::string_view text, const std::string_view source,
               const ParseParameters& parseParameters, const bool manifest) {
  auto parsed = envelope(text, source, manifest);
  if (mlir::failed(parsed)) {
    return mlir::failure();
  }
  const auto& root = (*parsed);
  if (mlir::failed(
          requireBenchmark(root, BenchmarkMetadata<Benchmark>::id, source))) {
    return mlir::failure();
  }
  auto benchmark = parseParameters(root["parameters"], source);
  if (mlir::failed(benchmark)) {
    return mlir::failure();
  }
  if (manifest) {
    const auto expected = manifestJSON((*benchmark));
    if (root.dump() != expected.dump()) {
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

template <class Benchmark>
[[nodiscard]] mlir::FailureOr<std::string>
evaluateBenchmark(const Benchmark& benchmark, const Counts& counts) {
  auto evaluation = benchmark.evaluate(counts);
  if (mlir::failed(evaluation)) {
    return mlir::failure();
  }
  const auto id = caseId(benchmark);
  /// Evaluation validates the total before this sum.
  const auto shots = std::accumulate(
      counts.begin(), counts.end(), size_t{0},
      [](const size_t sum, const auto& item) { return sum + item.second; });
  return evaluationToJSON(id, shots, (*evaluation));
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  mlir::FailureOr<std::string> evaluate##TYPE(const std::string_view manifest, \
                                              const std::string_view source,   \
                                              const Counts& counts) {          \
    auto result = STEM##FromManifestJSON(manifest, source);                    \
    if (mlir::failed(result)) {                                                \
      return mlir::failure();                                                  \
    }                                                                          \
    return evaluateBenchmark((*result), counts);                               \
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

mlir::FailureOr<std::string>
benchmarkIdFromInstanceSpecificationJSON(const std::string_view json,
                                         const std::string_view source) {
  auto result = envelope(json, source, false);
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  return (*result)["benchmark"].get<std::string>();
}

mlir::FailureOr<std::string>
benchmarkIdFromManifestJSON(const std::string_view json,
                            const std::string_view source) {
  auto result = envelope(json, source, true);
  if (mlir::failed(result)) {
    return mlir::failure();
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

mlir::FailureOr<std::string>
describeBenchmarkJSON(const std::string_view benchmark) {
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry->instanceSpecificationSchema().dump();
  }
  return ::mqt::emitError("unsupported benchmark '" + std::string(benchmark) +
                              "'",
                          ::mqt::ErrorCategory::InvalidArgument);
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  mlir::FailureOr<TYPE> STEM##FromInstanceSpecificationJSON(                   \
      const std::string_view json, const std::string_view source) {            \
    return parseBenchmark<TYPE>(json, source, parse##TYPE##Parameters, false); \
  }                                                                            \
  std::string toInstanceSpecificationJSON(const TYPE& benchmark) {             \
    return instanceSpecificationJSON(benchmark).dump();                        \
  }                                                                            \
  mlir::FailureOr<TYPE> STEM##FromManifestJSON(                                \
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

mlir::FailureOr<Counts> countsFromJSON(const std::string_view json,
                                       const std::string_view source) {
  auto parsed = parseJSON(json, source);
  if (mlir::failed(parsed)) {
    return mlir::failure();
  }
  const auto& root = (*parsed);
  if (mlir::failed(requireObject(root, source, "$"))) {
    return mlir::failure();
  }
  if (mlir::failed(
          rejectUnknownKeys(root, {"schema_version", "counts"}, source, "$"))) {
    return mlir::failure();
  }
  if (mlir::failed(requireSchemaVersion(root, source))) {
    return mlir::failure();
  }
  auto field = required(root, "counts", source, "$");
  if (mlir::failed(field)) {
    return mlir::failure();
  }
  const auto& values = *(*field);
  if (mlir::failed(requireObject(values, source, "$/counts"))) {
    return mlir::failure();
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
    if (mlir::failed(countResult)) {
      return mlir::failure();
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

mlir::FailureOr<std::string> evaluateJSON(const std::string_view manifest,
                                          const std::string_view counts,
                                          const std::string_view manifestSource,
                                          const std::string_view countsSource) {
  auto id = benchmarkIdFromManifestJSON(manifest, manifestSource);
  if (mlir::failed(id)) {
    return mlir::failure();
  }
  auto parsedCounts = countsFromJSON(counts, countsSource);
  if (mlir::failed(parsedCounts)) {
    return mlir::failure();
  }
  return findBenchmark((*id))->evaluate(manifest, manifestSource,
                                        (*parsedCounts));
}

mlir::FailureOr<std::string>
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

mlir::FailureOr<std::string>
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

mlir::FailureOr<ParsedBenchmark>
parseInstanceSpecificationJSON(const std::string_view json,
                               const std::string_view source) {
  auto id = benchmarkIdFromInstanceSpecificationJSON(json, source);
  if (mlir::failed(id)) {
    return mlir::failure();
  }
  auto instance = findBenchmark((*id))->parse(json, source);
  if (mlir::failed(instance)) {
    return mlir::failure();
  }
  return std::visit(
      [&](auto&& benchmark) {
        auto caseIdValue = caseId(benchmark);
        auto manifest = toManifestJSON(benchmark);
        return ParsedBenchmark{
            .instance = std::forward<decltype(benchmark)>(benchmark),
            .benchmarkId = (*std::move(id)),
            .caseId = std::move(caseIdValue),
            .manifestJSON = std::move(manifest),
        };
      },
      (*std::move(instance)));
}

} // namespace mqt::bench
