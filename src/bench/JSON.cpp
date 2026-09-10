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
#include "bench/Teleportation.hpp"

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
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
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
  [[nodiscard]] std::string evaluate##TYPE(std::string_view manifest,          \
                                           std::string_view source,            \
                                           const Counts& counts);
#include "bench/BenchmarkFamilies.inc"

using InstanceSpecificationSchemaFunction = Json (*)();
using EvaluationFunction = std::string (*)(std::string_view, std::string_view,
                                           const Counts&);

struct RegistryEntry {
  std::string_view id;
  uint64_t definitionVersion;
  InstanceSpecificationSchemaFunction instanceSpecificationSchema;
  EvaluationFunction evaluate;
};
constexpr std::array REGISTRY{
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  RegistryEntry{.id = (ID),                                                    \
                .definitionVersion = (DEFINITION_VERSION),                     \
                .instanceSpecificationSchema =                                 \
                    STEM##InstanceSpecificationSchema,                         \
                .evaluate = evaluate##TYPE},
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

[[noreturn]] void fail(const std::string_view source,
                       const std::string_view pointer,
                       const std::string_view message) {
  throw std::invalid_argument(std::string(source) + ":" + std::string(pointer) +
                              " " + std::string(message));
}

template <class Factory>
[[nodiscard]] auto constructBenchmark(const std::string_view source,
                                      const Factory& factory) {
  try {
    return factory();
  } catch (const std::invalid_argument& error) {
    fail(source, "$/parameters", error.what());
  }
}

[[nodiscard]] Json parseJSON(const std::string_view text,
                             const std::string_view source) {
  std::vector<std::unordered_set<std::string>> keysByDepth;
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
          if (!keys.emplace(key).second) {
            fail(source, "$", "contains duplicate key '" + key + "'");
          }
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
    fail(source, pointer, "must be encoded as an integer");
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

[[nodiscard]] const RegistryEntry&
requireBenchmarkEntry(const Json& root, const std::string_view source) {
  const auto benchmark = stringValue(required(root, "benchmark", source, "$"),
                                     source, "$/benchmark");
  if (const auto* entry = findBenchmark(benchmark)) {
    return *entry;
  }
  fail(source, "$/benchmark",
       "selects unsupported benchmark '" + benchmark + "'");
}

[[nodiscard]] Json
instanceSpecificationEnvelope(const std::string_view text,
                              const std::string_view source) {
  auto root = parseJSON(text, source);
  requireObject(root, source, "$");
  rejectUnknownKeys(root, {"schema_version", "benchmark", "parameters"}, source,
                    "$");
  requireSchemaVersion(root, source);
  static_cast<void>(requireBenchmarkEntry(root, source));
  requireObject(required(root, "parameters", source, "$"), source,
                "$/parameters");
  return root;
}

[[nodiscard]] Json manifestEnvelope(const std::string_view text,
                                    const std::string_view source) {
  auto root = parseJSON(text, source);
  requireObject(root, source, "$");
  rejectUnknownKeys(root,
                    {
                        "schema_version",
                        "case_id",
                        "benchmark",
                        "definition_version",
                        "parameters",
                        "outputs",
                        "reference",
                    },
                    source, "$");
  requireSchemaVersion(root, source);
  const auto& benchmark = requireBenchmarkEntry(root, source);
  const auto definition =
      unsignedInteger(required(root, "definition_version", source, "$"), source,
                      "$/definition_version");
  const auto expectedDefinition = benchmark.definitionVersion;
  if (definition != expectedDefinition) {
    fail(source, "$/definition_version",
         "must be " + std::to_string(expectedDefinition));
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

[[nodiscard]] BV parseBVParameters(const Json& parameters,
                                   const std::string_view source) {
  rejectUnknownKeys(parameters, {"hidden_bitstring", "method"}, source,
                    "$/parameters");
  BVOptions options{
      .hiddenBitstring = stringValue(
          required(parameters, "hidden_bitstring", source, "$/parameters"),
          source, "$/parameters/hidden_bitstring"),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    const auto value = stringValue(*method, source, "$/parameters/method");
    if (value == "static") {
      options.method = BVMethod::Static;
    } else if (value == "dynamic") {
      options.method = BVMethod::Dynamic;
    } else {
      fail(source, "$/parameters/method", "must be 'static' or 'dynamic'");
    }
  }
  return constructBenchmark(source,
                            [&options] { return BV(std::move(options)); });
}

[[nodiscard]] ModularMultiplier
parseModularMultiplierParameters(const Json& parameters,
                                 const std::string_view source) {
  rejectUnknownKeys(parameters,
                    {"multiplier", "modulus", "multiplicand", "control"},
                    source, "$/parameters");
  auto control = std::string("1");
  if (const auto value = parameters.find("control");
      value != parameters.end()) {
    control = stringValue(*value, source, "$/parameters/control");
  }
  if (control.size() != 1U) {
    fail(source, "$/parameters/control", "must be '0', '1', or '+'");
  }
  return constructBenchmark(source, [&] {
    return ModularMultiplier({
        .multiplier = stringValue(
            required(parameters, "multiplier", source, "$/parameters"), source,
            "$/parameters/multiplier"),
        .modulus =
            stringValue(required(parameters, "modulus", source, "$/parameters"),
                        source, "$/parameters/modulus"),
        .multiplicand = stringValue(
            required(parameters, "multiplicand", source, "$/parameters"),
            source, "$/parameters/multiplicand"),
        .control = control.front(),
    });
  });
}

[[nodiscard]] GHZ parseGHZParameters(const Json& parameters,
                                     const std::string_view source) {
  rejectUnknownKeys(parameters, {"qubits", "topology", "basis"}, source,
                    "$/parameters");
  GHZOptions options{
      .qubits =
          sizeValue(required(parameters, "qubits", source, "$/parameters"),
                    source, "$/parameters/qubits"),
  };
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
  return constructBenchmark(source, [&options] { return GHZ(options); });
}

[[nodiscard]] Grover parseGroverParameters(const Json& parameters,
                                           const std::string_view source) {
  rejectUnknownKeys(parameters, {"marked_bitstring", "iterations"}, source,
                    "$/parameters");
  GroverOptions options{
      .markedBitstring = stringValue(
          required(parameters, "marked_bitstring", source, "$/parameters"),
          source, "$/parameters/marked_bitstring"),
  };
  if (const auto iterations = parameters.find("iterations");
      iterations != parameters.end()) {
    options.iterations =
        sizeValue(*iterations, source, "$/parameters/iterations");
  }
  return constructBenchmark(source,
                            [&options] { return Grover(std::move(options)); });
}

[[nodiscard]] Multiplexer
parseMultiplexerParameters(const Json& parameters,
                           const std::string_view source) {
  rejectUnknownKeys(parameters, {"qubits"}, source, "$/parameters");
  return constructBenchmark(source, [&] {
    return Multiplexer({
        .qubits =
            sizeValue(required(parameters, "qubits", source, "$/parameters"),
                      source, "$/parameters/qubits"),
    });
  });
}

[[nodiscard]] QFT parseQFTParameters(const Json& parameters,
                                     const std::string_view source) {
  rejectUnknownKeys(parameters, {"qubits", "period_exponent", "method"}, source,
                    "$/parameters");
  QFTOptions options{
      .qubits =
          sizeValue(required(parameters, "qubits", source, "$/parameters"),
                    source, "$/parameters/qubits"),
      .periodExponent = sizeValue(
          required(parameters, "period_exponent", source, "$/parameters"),
          source, "$/parameters/period_exponent"),
  };
  if (const auto method = parameters.find("method");
      method != parameters.end()) {
    const auto value = stringValue(*method, source, "$/parameters/method");
    if (value == "standard") {
      options.method = QFTMethod::Standard;
    } else if (value == "semiclassical") {
      options.method = QFTMethod::Semiclassical;
    } else {
      fail(source, "$/parameters/method",
           "must be 'standard' or 'semiclassical'");
    }
  }
  return constructBenchmark(source, [&options] { return QFT(options); });
}

[[nodiscard]] QFTAdder parseQFTAdderParameters(const Json& parameters,
                                               const std::string_view source) {
  rejectUnknownKeys(parameters, {"addend", "accumulator", "method", "overflow"},
                    source, "$/parameters");
  QFTAdderOptions options{
      .addend =
          stringValue(required(parameters, "addend", source, "$/parameters"),
                      source, "$/parameters/addend"),
      .accumulator = stringValue(
          required(parameters, "accumulator", source, "$/parameters"), source,
          "$/parameters/accumulator"),
  };
  if (const auto it = parameters.find("method"); it != parameters.end()) {
    const auto value = stringValue(*it, source, "$/parameters/method");
    if (value == "register") {
      options.method = QFTAdderMethod::Register;
    } else if (value == "constant") {
      options.method = QFTAdderMethod::Constant;
    } else {
      fail(source, "$/parameters/method", "must be 'register' or 'constant'");
    }
  }
  if (const auto it = parameters.find("overflow"); it != parameters.end()) {
    const auto value = stringValue(*it, source, "$/parameters/overflow");
    if (value == "wrap") {
      options.overflow = QFTAdderOverflow::Wrap;
    } else if (value == "carry") {
      options.overflow = QFTAdderOverflow::Carry;
    } else {
      fail(source, "$/parameters/overflow", "must be 'wrap' or 'carry'");
    }
  }
  return constructBenchmark(
      source, [&options] { return QFTAdder(std::move(options)); });
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
  return constructBenchmark(source, [&] {
    return QPE({
        .precision = precision,
        .phase = Phase(numerator, denominator),
        .method = method,
    });
  });
}

[[nodiscard]] RepeatUntilSuccess
parseRepeatUntilSuccessParameters(const Json& parameters,
                                  const std::string_view source) {
  rejectUnknownKeys(parameters, {"data_qubits"}, source, "$/parameters");
  RepeatUntilSuccessOptions options;
  if (const auto width = parameters.find("data_qubits");
      width != parameters.end()) {
    options.dataQubits = sizeValue(*width, source, "$/parameters/data_qubits");
  }
  return constructBenchmark(source,
                            [&options] { return RepeatUntilSuccess(options); });
}

[[nodiscard]] Teleportation
parseTeleportationParameters(const Json& parameters,
                             const std::string_view source) {
  rejectUnknownKeys(parameters, {}, source, "$/parameters");
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

template <class Benchmark>
[[nodiscard]] Json instanceSpecificationJSON(const Benchmark& benchmark) {
  return {
      {"benchmark", std::string(BenchmarkMetadata<Benchmark>::id)},
      {"parameters", parametersJSON(benchmark)},
      {"schema_version", SCHEMA_VERSION},
  };
}

template <class Benchmark, class ParseParameters>
[[nodiscard]] Benchmark
parseInstanceSpecification(const std::string_view text,
                           const std::string_view source,
                           const ParseParameters& parseParameters) {
  const auto root = instanceSpecificationEnvelope(text, source);
  requireBenchmark(root, BenchmarkMetadata<Benchmark>::id, source);
  return parseParameters(root.at("parameters"), source);
}

template <class Benchmark, class ParseParameters>
[[nodiscard]] Benchmark parseManifest(const std::string_view text,
                                      const std::string_view source,
                                      const ParseParameters& parseParameters) {
  const auto root = manifestEnvelope(text, source);
  requireBenchmark(root, BenchmarkMetadata<Benchmark>::id, source);
  auto benchmark = parseParameters(root.at("parameters"), source);
  if (root.dump() != manifestJSON(benchmark).dump()) {
    fail(source, "$",
         "does not match its resolved benchmark instance and case ID");
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
[[nodiscard]] std::string evaluateBenchmark(const Benchmark& benchmark,
                                            const Counts& counts) {
  const auto shots = std::accumulate(
      counts.begin(), counts.end(), size_t{0},
      [](const size_t sum, const auto& item) { return sum + item.second; });
  return evaluationToJSON(caseId(benchmark), shots, benchmark.evaluate(counts));
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  std::string evaluate##TYPE(const std::string_view manifest,                  \
                             const std::string_view source,                    \
                             const Counts& counts) {                           \
    return evaluateBenchmark(STEM##FromManifestJSON(manifest, source),         \
                             counts);                                          \
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

std::string
benchmarkIdFromInstanceSpecificationJSON(const std::string_view json,
                                         const std::string_view source) {
  return instanceSpecificationEnvelope(json, source)
      .at("benchmark")
      .get<std::string>();
}

std::string benchmarkIdFromManifestJSON(const std::string_view json,
                                        const std::string_view source) {
  return manifestEnvelope(json, source).at("benchmark").get<std::string>();
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

std::string describeBenchmarkJSON(const std::string_view benchmark) {
  if (const auto* entry = findBenchmark(benchmark)) {
    return entry->instanceSpecificationSchema().dump();
  }
  throw std::invalid_argument("unsupported benchmark '" +
                              std::string(benchmark) + "'");
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  TYPE STEM##FromInstanceSpecificationJSON(const std::string_view json,        \
                                           const std::string_view source) {    \
    return parseInstanceSpecification<TYPE>(json, source,                      \
                                            parse##TYPE##Parameters);          \
  }                                                                            \
  std::string toInstanceSpecificationJSON(const TYPE& benchmark) {             \
    return instanceSpecificationJSON(benchmark).dump();                        \
  }                                                                            \
  TYPE STEM##FromManifestJSON(const std::string_view json,                     \
                              const std::string_view source) {                 \
    return parseManifest<TYPE>(json, source, parse##TYPE##Parameters);         \
  }                                                                            \
  std::string toManifestJSON(const TYPE& benchmark) {                          \
    return manifestJSON(benchmark).dump();                                     \
  }                                                                            \
  std::string caseId(const TYPE& benchmark) {                                  \
    return semanticCaseId(semanticJSON(benchmark));                            \
  }
#include "bench/BenchmarkFamilies.inc"

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

std::string evaluateJSON(const std::string_view manifest,
                         const std::string_view counts,
                         const std::string_view manifestSource,
                         const std::string_view countsSource) {
  const auto id = benchmarkIdFromManifestJSON(manifest, manifestSource);
  const auto parsedCounts = countsFromJSON(counts, countsSource);
  return findBenchmark(id)->evaluate(manifest, manifestSource, parsedCounts);
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

} // namespace mqt::bench
