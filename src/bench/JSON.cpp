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
#include "bench/FanOut.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/MagicStateDistillation.hpp"
#include "bench/ModularMultiplier.hpp"
#include "bench/Multiplexer.hpp"
#include "bench/QAOA.hpp"
#include "bench/QFT.hpp"
#include "bench/QFTAdder.hpp"
#include "bench/QPE.hpp"
#include "bench/RepeatUntilSuccess.hpp"
#include "bench/Shor.hpp"
#include "bench/Teleportation.hpp"
#include "bench/WState.hpp"
#include "bench/GroverWeakMeasurement.hpp"

#include "support/JSON.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

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
#include <utility>
#include <variant>

namespace mqt::bench {
namespace {

using Json = llvm::json::Value;

constexpr uint64_t SCHEMA_VERSION = 1;
constexpr uint64_t MANIFEST_VERSION = 2;
constexpr std::string_view CASE_DOMAIN = "mqt-core:benchmark-case:v2";

template <class Benchmark> struct BenchmarkMetadata;

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  template <> struct BenchmarkMetadata<TYPE> {                                 \
    static constexpr std::string_view id = ID;                                 \
    static constexpr uint64_t definitionVersion = DEFINITION_VERSION;          \
  };                                                                           \
  [[nodiscard]] Json STEM##InstanceSpecificationSchema();                      \
  [[nodiscard]] llvm::Expected<BenchmarkInstance> parse##TYPE##Instance(       \
      const Json& parameters, std::string_view source);
#include "bench/BenchmarkFamilies.inc"

using InstanceSpecificationSchemaFunction = Json (*)();
struct RegistryEntry {
  std::string_view id;
  uint64_t definitionVersion;
  InstanceSpecificationSchemaFunction instanceSpecificationSchema;
  llvm::Expected<BenchmarkInstance> (*parse)(const Json&, std::string_view);
};
constexpr std::array REGISTRY{
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  RegistryEntry{.id = (ID),                                                    \
                .definitionVersion = (DEFINITION_VERSION),                     \
                .instanceSpecificationSchema =                                 \
                    STEM##InstanceSpecificationSchema,                         \
                .parse = parse##TYPE##Instance},
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

[[nodiscard]] llvm::Error fail(const std::string_view source,
                               const std::string_view pointer,
                               const std::string_view message) {
  return llvm::createStringError(llvm::StringRef(source) + ":" +
                                 llvm::StringRef(pointer) + " " +
                                 llvm::StringRef(message));
}

bool keys(const Json& value, std::initializer_list<llvm::StringRef> known,
          llvm::json::Path path) {
  const auto* object = value.getAsObject();
  if (object == nullptr) {
    path.report("must be an object");
    return false;
  }
  return std::ranges::all_of(*object, [&](const auto& field) {
    const llvm::StringRef key = field.first;
    if (std::ranges::find(known, key) != known.end()) {
      return true;
    }
    path.field(key).report("unknown key");
    return false;
  });
}

template <class Factory>
[[nodiscard]] auto constructBenchmark(const std::string_view source,
                                      const Factory& factory)
    -> llvm::Expected<decltype(factory())> {
  try {
    return factory();
  } catch (const std::invalid_argument& error) {
    return fail(source, "$/parameters", error.what());
  }
}

template <class Enum>
bool enumValue(const Json& parameters, llvm::StringRef key, Enum& out,
               std::initializer_list<std::pair<llvm::StringRef, Enum>> values,
               llvm::json::Path path) {
  const auto* field = parameters.getAsObject()->get(key);
  if (!field) {
    return true;
  }
  if (const auto name = field->getAsString()) {
    for (const auto& [spelling, value] : values) {
      if (*name == spelling) {
        out = value;
        return true;
      }
    }
  }
  path.field(key).report("has an unsupported value");
  return false;
}

[[nodiscard]] llvm::Expected<Json> envelope(const std::string_view text,
                                            const std::string_view source,
                                            const bool manifest) {
  auto root = mqt::json::parse(text, source);
  if (!root) {
    return root.takeError();
  }
  llvm::json::Path::Root diagnostic(source);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(*root, path);
  uint64_t version = 0;
  std::string benchmark;
  if (!object ||
      !(manifest ? keys(*root,
                        {
                            "schema_version",
                            "benchmark",
                            "parameters",
                            "case_id",
                            "definition_version",
                            "outputs",
                            "reference",
                        },
                        path)
                 : keys(*root, {"schema_version", "benchmark", "parameters"},
                        path)) ||
      !object.map("schema_version", version) ||
      !object.map("benchmark", benchmark)) {
    return diagnostic.getError();
  }
  if (version != (manifest ? MANIFEST_VERSION : SCHEMA_VERSION)) {
    return fail(source, "$/schema_version",
                manifest ? "must be 2; regenerate version 1 manifests from "
                           "their instance specifications"
                         : "must be 1");
  }
  const auto* entry = findBenchmark(benchmark);
  if (entry == nullptr) {
    return fail(source, "$/benchmark",
                "selects unsupported benchmark '" + benchmark + "'");
  }
  const auto* parameters = root->getAsObject()->getObject("parameters");
  if (parameters == nullptr) {
    return fail(source, "$/parameters", "is required and must be an object");
  }
  if (manifest) {
    uint64_t definition = 0;
    std::string caseID;
    if (!object.map("definition_version", definition) ||
        !object.map("case_id", caseID)) {
      return diagnostic.getError();
    }
    if (definition != entry->definitionVersion) {
      return fail(source, "$/definition_version",
                  "must be " + std::to_string(entry->definitionVersion));
    }
    if (root->getAsObject()->getArray("outputs") == nullptr) {
      return fail(source, "$/outputs", "is required and must be an array");
    }
    if (root->getAsObject()->getObject("reference") == nullptr) {
      return fail(source, "$/reference", "is required and must be an object");
    }
  }
  return root;
}

[[nodiscard]] llvm::Expected<BV> parseBVParameters(const Json& parameters,
                                                   std::string_view source) {
  BVOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"hidden_bitstring", "method"}, path) ||
      !object.map("hidden_bitstring", options.hiddenBitstring) ||
      !enumValue<BVMethod>(
          parameters, "method", options.method,
          {{"static", BVMethod::Static}, {"dynamic", BVMethod::Dynamic}},
          path)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return BV(std::move(options)); });
}

[[nodiscard]] llvm::Expected<FanOut>
parseFanOutParameters(const Json& parameters, std::string_view source) {
  FanOutOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"controls"}, path) ||
      !object.map("controls", options.controls)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return FanOut(options); });
}

[[nodiscard]] llvm::Expected<GHZ> parseGHZParameters(const Json& parameters,
                                                     std::string_view source) {
  GHZOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"qubits", "topology", "basis"}, path) ||
      !object.map("qubits", options.qubits) ||
      !enumValue<GHZTopology>(
          parameters, "topology", options.topology,
          {{"linear", GHZTopology::Linear}, {"star", GHZTopology::Star}},
          path) ||
      !enumValue<GHZBasis>(parameters, "basis", options.basis,
                           {{"z", GHZBasis::Z}, {"x", GHZBasis::X}}, path)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return GHZ(options); });
}

[[nodiscard]] llvm::Expected<Grover>
parseGroverParameters(const Json& parameters, std::string_view source) {
  GroverOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  size_t iterations = 0;
  if (!object || !keys(parameters, {"marked_bitstring", "iterations"}, path) ||
      !object.map("marked_bitstring", options.markedBitstring) ||
      !object.mapOptional("iterations", iterations)) {
    return diagnostic.getError();
  }
  if (parameters.getAsObject()->get("iterations") != nullptr) {
    options.iterations = iterations;
  }
  return constructBenchmark(source, [&] { return Grover(std::move(options)); });
}

[[nodiscard]] llvm::Expected<GroverWeakMeasurement>
parseGroverWeakMeasurementParameters(const Json& parameters,
                                     std::string_view source) {
  GroverWeakMeasurementOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  double measurementStrength = 0;
  if (!object ||
      !keys(parameters, {"marked_bitstring", "measurement_strength"}, path) ||
      !object.map("marked_bitstring", options.markedBitstring) ||
      !object.mapOptional("measurement_strength", measurementStrength)) {
    return diagnostic.getError();
  }
  if (parameters.getAsObject()->get("measurement_strength") != nullptr) {
    options.measurementStrength = measurementStrength;
  }
  return constructBenchmark(
      source, [&] { return GroverWeakMeasurement(std::move(options)); });
}

[[nodiscard]] llvm::Expected<MagicStateDistillation>
parseMagicStateDistillationParameters(const Json& parameters,
                                      std::string_view source) {
  MagicStateDistillationOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"levels"}, path) ||
      !object.mapOptional("levels", options.levels)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source,
                            [&] { return MagicStateDistillation(options); });
}

[[nodiscard]] llvm::Expected<ModularMultiplier>
parseModularMultiplierParameters(const Json& parameters,
                                 std::string_view source) {
  ModularMultiplierOptions options{};
  std::string control = "1";
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object ||
      !keys(parameters, {"multiplier", "modulus", "multiplicand", "control"},
            path) ||
      !object.map("multiplier", options.multiplier) ||
      !object.map("modulus", options.modulus) ||
      !object.map("multiplicand", options.multiplicand) ||
      !object.mapOptional("control", control)) {
    return diagnostic.getError();
  }
  if (control.size() != 1) {
    return fail(source, "$/parameters/control", "must be 0, 1, or +");
  }
  options.control = control.front();
  return constructBenchmark(
      source, [&] { return ModularMultiplier(std::move(options)); });
}

[[nodiscard]] llvm::Expected<Multiplexer>
parseMultiplexerParameters(const Json& parameters, std::string_view source) {
  MultiplexerOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"qubits"}, path) ||
      !object.map("qubits", options.qubits)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return Multiplexer(options); });
}

[[nodiscard]] llvm::Expected<QAOA>
parseQAOAParameters(const Json& parameters, std::string_view source) {
  QAOAOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"qubits", "depth"}, path) ||
      !object.map("qubits", options.qubits) ||
      !object.mapOptional("depth", options.depth)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return QAOA(options); });
}

[[nodiscard]] llvm::Expected<QFT> parseQFTParameters(const Json& parameters,
                                                     std::string_view source) {
  QFTOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object ||
      !keys(parameters, {"qubits", "period_exponent", "method"}, path) ||
      !object.map("qubits", options.qubits) ||
      !object.map("period_exponent", options.periodExponent) ||
      !enumValue<QFTMethod>(parameters, "method", options.method,
                            {
                                {"standard", QFTMethod::Standard},
                                {"semiclassical", QFTMethod::Semiclassical},
                            },
                            path)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return QFT(options); });
}

[[nodiscard]] llvm::Expected<QFTAdder>
parseQFTAdderParameters(const Json& parameters, std::string_view source) {
  QFTAdderOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object ||
      !keys(parameters, {"addend", "accumulator", "method", "overflow"},
            path) ||
      !object.map("addend", options.addend) ||
      !object.map("accumulator", options.accumulator) ||
      !enumValue<QFTAdderMethod>(parameters, "method", options.method,
                                 {
                                     {"register", QFTAdderMethod::Register},
                                     {"constant", QFTAdderMethod::Constant},
                                 },
                                 path) ||
      !enumValue<QFTAdderOverflow>(parameters, "overflow", options.overflow,
                                   {
                                       {"wrap", QFTAdderOverflow::Wrap},
                                       {"carry", QFTAdderOverflow::Carry},
                                   },
                                   path)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source,
                            [&] { return QFTAdder(std::move(options)); });
}

[[nodiscard]] llvm::Expected<RepeatUntilSuccess>
parseRepeatUntilSuccessParameters(const Json& parameters,
                                  std::string_view source) {
  RepeatUntilSuccessOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"data_qubits"}, path) ||
      !object.mapOptional("data_qubits", options.dataQubits)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source,
                            [&] { return RepeatUntilSuccess(options); });
}

[[nodiscard]] llvm::Expected<Shor>
parseShorParameters(const Json& parameters, std::string_view source) {
  ShorOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"number", "base"}, path) ||
      !object.map("number", options.number) ||
      !object.mapOptional("base", options.base)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return Shor(options); });
}

[[nodiscard]] llvm::Expected<WState>
parseWStateParameters(const Json& parameters, std::string_view source) {
  WStateOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  if (!object || !keys(parameters, {"qubits"}, path) ||
      !object.map("qubits", options.qubits)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] { return WState(options); });
}

[[nodiscard]] llvm::Expected<Teleportation>
parseTeleportationParameters(const Json& parameters, std::string_view source) {
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  if (!keys(parameters, {}, diagnostic)) {
    return diagnostic.getError();
  }
  return Teleportation{};
}

[[nodiscard]] llvm::Expected<QPE> parseQPEParameters(const Json& parameters,
                                                     std::string_view source) {
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  llvm::json::ObjectMapper object(parameters, path);
  size_t precision = 0;
  auto method = QPEMethod::Standard;
  if (!object || !keys(parameters, {"precision", "phase", "method"}, path) ||
      !object.map("precision", precision) ||
      !enumValue<QPEMethod>(parameters, "method", method,
                            {
                                {"standard", QPEMethod::Standard},
                                {"iterative", QPEMethod::Iterative},
                            },
                            path)) {
    return diagnostic.getError();
  }
  const auto* phase = parameters.getAsObject()->get("phase");
  if (phase == nullptr) {
    return fail(source, "$/parameters/phase", "is required");
  }
  llvm::json::ObjectMapper phaseObject(*phase, path.field("phase"));
  uint64_t numerator = 0;
  uint64_t denominator = 0;
  if (!phaseObject ||
      !keys(*phase, {"numerator", "denominator"}, path.field("phase")) ||
      !phaseObject.map("numerator", numerator) ||
      !phaseObject.map("denominator", denominator)) {
    return diagnostic.getError();
  }
  return constructBenchmark(source, [&] {
    return QPE({
        .precision = precision,
        .phase = Phase(numerator, denominator),
        .method = method,
    });
  });
}

[[nodiscard]] llvm::StringRef topologyName(const GHZTopology topology) {
  return topology == GHZTopology::Linear ? "linear" : "star";
}

[[nodiscard]] llvm::StringRef basisName(const GHZBasis basis) {
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

[[nodiscard]] Json parametersJSON(const Shor& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "number", .V = options.number},
      {.K = "base", .V = options.base},
  };
}

[[nodiscard]] Json parametersJSON(const BV& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "hidden_bitstring", .V = options.hiddenBitstring},
      {.K = "method", .V = methodName(options.method)},
  };
}

[[nodiscard]] Json parametersJSON(const FanOut& benchmark) {
  return llvm::json::Object{
      {.K = "controls", .V = benchmark.options().controls},
  };
}

[[nodiscard]] Json parametersJSON(const GHZ& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "basis", .V = basisName(options.basis)},
      {.K = "qubits", .V = options.qubits},
      {.K = "topology", .V = topologyName(options.topology)},
  };
}

[[nodiscard]] Json parametersJSON(const Grover& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "iterations", .V = *options.iterations},
      {.K = "marked_bitstring", .V = options.markedBitstring},
  };
}

[[nodiscard]] Json parametersJSON(const GroverWeakMeasurement& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "marked_bitstring", .V = options.markedBitstring},
      {.K = "measurement_strength", .V = *options.measurementStrength},
  };
}

[[nodiscard]] Json parametersJSON(const MagicStateDistillation& benchmark) {
  return llvm::json::Object{{.K = "levels", .V = benchmark.options().levels}};
}

[[nodiscard]] Json parametersJSON(const ModularMultiplier& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "control", .V = std::string(1, options.control)},
      {.K = "multiplicand", .V = options.multiplicand},
      {.K = "modulus", .V = options.modulus},
      {.K = "multiplier", .V = options.multiplier},
  };
}

[[nodiscard]] Json parametersJSON(const Multiplexer& benchmark) {
  return llvm::json::Object{{.K = "qubits", .V = benchmark.options().qubits}};
}

[[nodiscard]] Json parametersJSON(const QAOA& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "depth", .V = options.depth},
      {.K = "qubits", .V = options.qubits},
  };
}

[[nodiscard]] Json parametersJSON(const QFT& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "method", .V = methodName(options.method)},
      {.K = "period_exponent", .V = options.periodExponent},
      {.K = "qubits", .V = options.qubits},
  };
}

[[nodiscard]] Json parametersJSON(const QFTAdder& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "addend", .V = options.addend},
      {.K = "accumulator", .V = options.accumulator},
      {
          .K = "method",
          .V = options.method == QFTAdderMethod::Register ? "register"
                                                          : "constant",
      },
      {
          .K = "overflow",
          .V = options.overflow == QFTAdderOverflow::Wrap ? "wrap" : "carry",
      },
  };
}

[[nodiscard]] Json parametersJSON(const QPE& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {.K = "method", .V = methodName(options.method)},
      {
          .K = "phase",
          .V =
              llvm::json::Object{
                  {.K = "denominator", .V = options.phase.denominator()},
                  {.K = "numerator", .V = options.phase.numerator()},
              },
      },
      {.K = "precision", .V = options.precision},
  };
}

[[nodiscard]] Json parametersJSON(const RepeatUntilSuccess& benchmark) {
  return llvm::json::Object{
      {.K = "data_qubits", .V = benchmark.options().dataQubits},
  };
}

[[nodiscard]] Json parametersJSON(const Teleportation& /*unused*/) {
  return llvm::json::Object{};
}

[[nodiscard]] Json parametersJSON(const WState& benchmark) {
  return llvm::json::Object{{.K = "qubits", .V = benchmark.options().qubits}};
}

[[nodiscard]] Json analyticReferenceJSON(
    const Output& output, const std::string_view model,
    const std::optional<std::string_view> successOutcome = std::nullopt) {
  llvm::json::Object reference = llvm::json::Object{
      {.K = "kind", .V = "analytic"},
      {.K = "model", .V = std::string(model)},
      {.K = "outcome_order", .V = "big_endian"},
      {.K = "output", .V = output.name},
      {.K = "version", .V = uint64_t{1}},
  };
  if (successOutcome) {
    reference["success_outcome"] = std::string(*successOutcome);
  }
  return reference;
}

[[nodiscard]] Json referenceJSON(const QAOA& benchmark) {
  return llvm::json::Object{
      {.K = "kind", .V = "verification"},
      {.K = "model", .V = "qaoa_max_cut_cycle"},
      {.K = "outcome_order", .V = "big_endian"},
      {.K = "output", .V = benchmark.output().name},
      {.K = "version", .V = uint64_t{1}},
  };
}

[[nodiscard]] Json referenceJSON(const Shor& benchmark) {
  return llvm::json::Object{
      {.K = "kind", .V = "verification"},
      {.K = "model", .V = "shor_factors"},
      {.K = "outcome_order", .V = "big_endian"},
      {.K = "output", .V = benchmark.output().name},
      {.K = "version", .V = uint64_t{1}},
  };
}

[[nodiscard]] Json referenceJSON(const BV& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "bernstein_vazirani",
                               benchmark.options().hiddenBitstring);
}

[[nodiscard]] Json referenceJSON(const FanOut& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "fan_out");
}

[[nodiscard]] Json referenceJSON(const GHZ& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "ghz");
}

[[nodiscard]] Json referenceJSON(const Grover& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "grover_single_marked",
                               benchmark.options().markedBitstring);
}

[[nodiscard]] Json referenceJSON(const GroverWeakMeasurement& benchmark) {
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
  return llvm::json::Object{
      {.K = "benchmark", .V = std::string(id)},
      {.K = "definition_version", .V = definitionVersionValue},
      {
          .K = "outputs",
          .V =
              llvm::json::Array{
                  llvm::json::Object{
                      {.K = "name", .V = output.name},
                      {.K = "width", .V = output.width},
                  },
              },
      },
      {.K = "parameters", .V = parameters},
      {.K = "reference", .V = reference},
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
  llvm::raw_string_ostream raw(input);
  llvm::json::OStream stream(raw);
  mqt::json::value(stream, semantic);
  return "sha256-" +
         llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(input)),
                     true);
}

template <class Benchmark>
[[nodiscard]] Json manifestJSON(const Benchmark& benchmark) {
  auto semantic = semanticJSON(benchmark);
  (*semantic.getAsObject())["case_id"] = semanticCaseId(semantic);
  (*semantic.getAsObject())["schema_version"] = MANIFEST_VERSION;
  return semantic;
}

template <class Benchmark>
[[nodiscard]] Json instanceSpecificationJSON(const Benchmark& benchmark) {
  return llvm::json::Object{
      {"benchmark", std::string(BenchmarkMetadata<Benchmark>::id)},
      {"parameters", parametersJSON(benchmark)},
      {"schema_version", SCHEMA_VERSION},
  };
}

template <class Benchmark>
[[nodiscard]] bool matchesManifest(const Json& root,
                                   const Benchmark& benchmark) {
  if (root != manifestJSON(benchmark)) {
    return false;
  }
  const auto& output = root.getAsObject()->getArray("outputs")->front();
  const auto& width = *output.getAsObject()->get("width");
  if (const auto integer = width.getAsUINT64()) {
    return *integer == benchmark.output().width;
  }
  // Equality and signed-64-bit output dimensions bound this double by 2^63.
  return static_cast<uint64_t>(*width.getAsNumber()) ==
         benchmark.output().width;
}

template <class Benchmark, class ParseParameters>
[[nodiscard]] llvm::Expected<Benchmark>
parseBenchmark(const std::string_view text, const std::string_view source,
               const bool manifest, const ParseParameters& parseParameters) {
  auto root = envelope(text, source, manifest);
  if (!root) {
    return root.takeError();
  }
  if (*root->getAsObject()->getString("benchmark") !=
      llvm::StringRef(BenchmarkMetadata<Benchmark>::id)) {
    return fail(source, "$/benchmark",
                "must be '" + std::string(BenchmarkMetadata<Benchmark>::id) +
                    "'");
  }
  auto benchmark =
      parseParameters(*root->getAsObject()->get("parameters"), source);
  if (!benchmark) {
    return benchmark.takeError();
  }
  if (manifest && !matchesManifest(*root, *benchmark)) {
    return fail(source, "$",
                "does not match its resolved benchmark instance and case ID");
  }
  return benchmark;
}

template <class Benchmark>
[[nodiscard]] Json
baseInstanceSpecificationSchema(llvm::json::Object parameters) {
  parameters["additionalProperties"] = false;
  parameters["type"] = "object";
  using Metadata = BenchmarkMetadata<Benchmark>;
  return llvm::json::Object{
      {"$schema", "https://json-schema.org/draft/2020-12/schema"},
      {"additionalProperties", false},
      {
          "properties",
          llvm::json::Object{
              {
                  .K = "benchmark",
                  .V =
                      llvm::json::Object{
                          {
                              .K = "const",
                              .V = std::string(Metadata::id),
                          },
                      },
              },
              {.K = "parameters", .V = std::move(parameters)},
              {
                  .K = "schema_version",
                  .V = llvm::json::Object{{.K = "const", .V = SCHEMA_VERSION}},
              },
          },
      },
      {
          "required",
          llvm::json::Array{"schema_version", "benchmark", "parameters"},
      },
      {"type", "object"},
      {"x-mqt-definition-version", Metadata::definitionVersion},
  };
}

[[nodiscard]] Json bvInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<BV>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "hidden_bitstring",
                      .V =
                          llvm::json::Object{
                              {.K = "maxLength", .V = BVOptions::MAX_BITS},
                              {.K = "minLength", .V = 1},
                              {.K = "pattern", .V = "^[01]+$"},
                              {.K = "type", .V = "string"},
                          },
                  },
                  {
                      .K = "method",
                      .V =
                          llvm::json::Object{
                              {.K = "default", .V = "static"},
                              {
                                  .K = "enum",
                                  .V = llvm::json::Array{"static", "dynamic"},
                              },
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"hidden_bitstring"}},
  });
}

[[nodiscard]] Json fanOutInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<FanOut>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "controls",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maximum",
                                  .V = FanOutOptions::MAX_CONTROLS,
                              },
                              {.K = "minimum", .V = 1},
                              {.K = "type", .V = "integer"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"controls"}},
  });
}

[[nodiscard]] Json ghzInstanceSpecificationSchema() {
  auto parameters =
      llvm::json::Object{
          {
              .K = "properties",
              .V =
                  llvm::json::Object{
                      {
                          .K = "basis",
                          .V =
                              llvm::json::Object{
                                  {.K = "default", .V = "z"},
                                  {
                                      .K = "enum",
                                      .V = llvm::json::Array{"z", "x"},
                                  },
                              },
                      },
                      {
                          .K = "qubits",
                          .V =
                              llvm::json::Object{
                                  {.K = "maximum", .V = GHZOptions::MAX_QUBITS},
                                  {.K = "minimum", .V = 1},
                                  {.K = "type", .V = "integer"},
                              },
                      },
                      {
                          .K = "topology",
                          .V =
                              llvm::json::Object{
                                  {.K = "default", .V = "linear"},
                                  {
                                      .K = "enum",
                                      .V = llvm::json::Array{"linear", "star"},
                                  },
                              },
                      },
                  },
          },
          {.K = "required", .V = llvm::json::Array{"qubits"}},
  };
  parameters["allOf"] =
      llvm::json::Array{
          llvm::json::Object{
              {
                  .K = "if",
                  .V =
                      llvm::json::Object{
                          {
                              .K = "properties",
                              .V =
                                  llvm::json::Object{
                                      {
                                          .K = "basis",
                                          .V =
                                              llvm::json::Object{
                                                  {
                                                      .K = "const",
                                                      .V = "x",
                                                  },
                                              },
                                      },
                                  },
                          },
                          {.K = "required", .V = llvm::json::Array{"basis"}},
                      },
              },
              {
                  .K = "then",
                  .V =
                      llvm::json::Object{
                          {
                              .K = "properties",
                              .V =
                                  llvm::json::Object{
                                      {
                                          .K = "qubits",
                                          .V =
                                              llvm::json::Object{
                                                  {
                                                      .K = "maximum",
                                                      .V =
                                                          GHZOptions::MAX_X_BASIS_QUBITS,
                                                  },
                                              },
                                      },
                                  },
                          },
                      },
              },
          },
  };
  return baseInstanceSpecificationSchema<GHZ>(std::move(parameters));
}

[[nodiscard]] Json groverInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Grover>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "iterations",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maximum",
                                  .V = std::numeric_limits<int32_t>::max(),
                              },
                              {.K = "minimum", .V = 0},
                              {.K = "type", .V = "integer"},
                          },
                  },
                  {
                      .K = "marked_bitstring",
                      .V =
                          llvm::json::Object{
                              {.K = "maxLength", .V = 62},
                              {.K = "minLength", .V = 2},
                              {.K = "pattern", .V = "^[01]+$"},
                              {.K = "type", .V = "string"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"marked_bitstring"}},
  });
}

[[nodiscard]] Json groverWeakMeasurementInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<
      GroverWeakMeasurement>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "marked_bitstring",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maxLength",
                                  .V = GroverWeakMeasurementOptions::MAX_QUBITS,
                              },
                              {.K = "minLength", .V = 2},
                              {.K = "pattern", .V = "^[01]+$"},
                              {.K = "type", .V = "string"},
                          },
                  },
                  {
                      .K = "measurement_strength",
                      .V =
                          llvm::json::Object{
                              {.K = "exclusiveMinimum", .V = 0},
                              {.K = "maximum", .V = 0.5},
                              {.K = "type", .V = "number"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"marked_bitstring"}},
  });
}

[[nodiscard]] Json magicStateDistillationInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<MagicStateDistillation>(
      llvm::json::Object{
          {
              .K = "properties",
              .V =
                  llvm::json::Object{
                      {
                          .K = "levels",
                          .V =
                              llvm::json::Object{
                                  {.K = "default", .V = 1},
                                  {.K = "minimum", .V = 1},
                                  {
                                      .K = "maximum",
                                      .V = std::numeric_limits<int64_t>::max() /
                                           5,
                                  },
                                  {.K = "type", .V = "integer"},
                              },
                      },
                  },
          },
      });
}

[[nodiscard]] Json modularMultiplierInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<ModularMultiplier>(
      llvm::json::Object{
          {
              .K = "properties",
              .V =
                  llvm::json::Object{
                      {
                          .K = "control",
                          .V =
                              llvm::json::Object{
                                  {.K = "default", .V = "1"},
                                  {
                                      .K = "enum",
                                      .V = llvm::json::Array{"0", "1", "+"},
                                  },
                              },
                      },
                      {
                          .K = "multiplicand",
                          .V =
                              llvm::json::Object{
                                  {.K = "type", .V = "string"},
                                  {.K = "minLength", .V = 2},
                                  {
                                      .K = "maxLength",
                                      .V = ModularMultiplierOptions::MAX_BITS,
                                  },
                                  {.K = "pattern", .V = "^[01+]+$"},
                              },
                      },
                      {
                          .K = "modulus",
                          .V =
                              llvm::json::Object{
                                  {
                                      .K = "maxLength",
                                      .V = ModularMultiplierOptions::MAX_BITS,
                                  },
                                  {.K = "minLength", .V = 2},
                                  {.K = "pattern", .V = "^1[01]+$"},
                                  {.K = "type", .V = "string"},
                              },
                      },
                      {
                          .K = "multiplier",
                          .V =
                              llvm::json::Object{
                                  {
                                      .K = "maxLength",
                                      .V = ModularMultiplierOptions::MAX_BITS,
                                  },
                                  {.K = "minLength", .V = 2},
                                  {.K = "pattern", .V = "^[01]+$"},
                                  {.K = "type", .V = "string"},
                              },
                      },
                  },
          },
          {
              .K = "required",
              .V = llvm::json::Array{"multiplier", "modulus", "multiplicand"},
          },
      });
}

[[nodiscard]] Json multiplexerInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Multiplexer>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "qubits",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maximum",
                                  .V = MultiplexerOptions::MAX_QUBITS,
                              },
                              {.K = "minimum", .V = 2},
                              {.K = "type", .V = "integer"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"qubits"}},
  });
}

[[nodiscard]] Json qaoaInstanceSpecificationSchema() {
  const QAOAOptions defaults{.qubits = 3};
  return baseInstanceSpecificationSchema<QAOA>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "depth",
                      .V =
                          llvm::json::Object{
                              {.K = "default", .V = defaults.depth},
                              {.K = "maximum", .V = QAOAOptions::MAX_DEPTH},
                              {.K = "minimum", .V = 1},
                              {.K = "type", .V = "integer"},
                          },
                  },
                  {
                      .K = "qubits",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maximum",
                                  .V = std::numeric_limits<int64_t>::max(),
                              },
                              {.K = "minimum", .V = 3},
                              {.K = "type", .V = "integer"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"qubits"}},
  });
}

[[nodiscard]] Json qftInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<QFT>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "method",
                      .V =
                          llvm::json::Object{
                              {.K = "default", .V = "standard"},
                              {
                                  .K = "enum",
                                  .V =
                                      llvm::json::Array{
                                          "standard",
                                          "semiclassical",
                                      },
                              },
                          },
                  },
                  {
                      .K = "period_exponent",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maximum",
                                  .V = QFTOptions::MAX_PERIOD_EXPONENT,
                              },
                              {.K = "minimum", .V = 0},
                              {.K = "type", .V = "integer"},
                          },
                  },
                  {
                      .K = "qubits",
                      .V =
                          llvm::json::Object{
                              {.K = "maximum", .V = QFTOptions::MAX_QUBITS},
                              {.K = "minimum", .V = 1},
                              {.K = "type", .V = "integer"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"qubits", "period_exponent"}},
  });
}

[[nodiscard]] Json qftAdderInstanceSpecificationSchema() {
  llvm::json::Object properties{
      {
          .K = "addend",
          .V =
              llvm::json::Object{
                  {.K = "type", .V = "string"},
                  {.K = "minLength", .V = 1},
                  {.K = "maxLength", .V = QFTAdderOptions::MAX_QUBITS},
                  {.K = "pattern", .V = "^[01+]+$"},
              },
      },
      {
          .K = "accumulator",
          .V =
              llvm::json::Object{
                  {.K = "type", .V = "string"},
                  {.K = "minLength", .V = 1},
                  {.K = "maxLength", .V = QFTAdderOptions::MAX_QUBITS},
                  {.K = "pattern", .V = "^[01]+$"},
              },
      },
      {
          .K = "method",
          .V =
              llvm::json::Object{
                  {.K = "type", .V = "string"},
                  {.K = "enum", .V = llvm::json::Array{"register", "constant"}},
                  {.K = "default", .V = "register"},
              },
      },
      {
          .K = "overflow",
          .V =
              llvm::json::Object{
                  {.K = "type", .V = "string"},
                  {.K = "enum", .V = llvm::json::Array{"wrap", "carry"}},
                  {.K = "default", .V = "wrap"},
              },
      },
  };
  llvm::json::Object constantCondition{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "method",
                      .V = llvm::json::Object{{.K = "const", .V = "constant"}},
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"method"}},
  };
  llvm::json::Object constantProperties{
      {
          .K = "addend",
          .V = llvm::json::Object{{.K = "pattern", .V = "^[01]+$"}},
      },
  };
  llvm::json::Object carryCondition{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "overflow",
                      .V = llvm::json::Object{{.K = "const", .V = "carry"}},
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"overflow"}},
  };
  llvm::json::Object carryProperties{
      {
          .K = "addend",
          .V =
              llvm::json::Object{
                  {.K = "maxLength", .V = QFTAdderOptions::MAX_QUBITS - 1U},
              },
      },
      {
          .K = "accumulator",
          .V =
              llvm::json::Object{
                  {.K = "maxLength", .V = QFTAdderOptions::MAX_QUBITS - 1U},
              },
      },
  };
  llvm::json::Array constraints{
      llvm::json::Object{
          {.K = "if", .V = std::move(constantCondition)},
          {
              .K = "then",
              .V =
                  llvm::json::Object{
                      {.K = "properties", .V = std::move(constantProperties)},
                  },
          },
      },
      llvm::json::Object{
          {.K = "if", .V = std::move(carryCondition)},
          {
              .K = "then",
              .V =
                  llvm::json::Object{
                      {.K = "properties", .V = std::move(carryProperties)},
                  },
          },
      },
  };
  return baseInstanceSpecificationSchema<QFTAdder>(llvm::json::Object{
      {.K = "properties", .V = std::move(properties)},
      {.K = "allOf", .V = std::move(constraints)},
      {.K = "required", .V = llvm::json::Array{"addend", "accumulator"}},
  });
}

[[nodiscard]] Json qpeInstanceSpecificationSchema() {
  llvm::json::Object phaseProperties{
      {
          .K = "denominator",
          .V =
              llvm::json::Object{
                  {.K = "maximum", .V = std::numeric_limits<uint64_t>::max()},
                  {.K = "minimum", .V = 1},
                  {.K = "type", .V = "integer"},
              },
      },
      {
          .K = "numerator",
          .V =
              llvm::json::Object{
                  {.K = "maximum", .V = std::numeric_limits<uint64_t>::max()},
                  {.K = "minimum", .V = 0},
                  {.K = "type", .V = "integer"},
              },
      },
  };
  llvm::json::Object phase{
      {.K = "additionalProperties", .V = false},
      {.K = "properties", .V = std::move(phaseProperties)},
      {.K = "required", .V = llvm::json::Array{"numerator", "denominator"}},
      {.K = "type", .V = "object"},
  };
  llvm::json::Object properties{
      {
          .K = "method",
          .V =
              llvm::json::Object{
                  {.K = "default", .V = "standard"},
                  {
                      .K = "enum",
                      .V = llvm::json::Array{"standard", "iterative"},
                  },
              },
      },
      {.K = "phase", .V = std::move(phase)},
      {
          .K = "precision",
          .V =
              llvm::json::Object{
                  {.K = "maximum", .V = QPEOptions::MAX_PRECISION},
                  {.K = "minimum", .V = 1},
                  {.K = "type", .V = "integer"},
              },
      },
  };
  return baseInstanceSpecificationSchema<QPE>(llvm::json::Object{
      {.K = "properties", .V = std::move(properties)},
      {.K = "required", .V = llvm::json::Array{"precision", "phase"}},
  });
}

[[nodiscard]] Json repeatUntilSuccessInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<RepeatUntilSuccess>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "data_qubits",
                      .V =
                          llvm::json::Object{
                              {.K = "default", .V = 1},
                              {.K = "minimum", .V = 1},
                              {
                                  .K = "maximum",
                                  .V = RepeatUntilSuccessOptions::
                                      MAX_DATA_QUBITS,
                              },
                              {.K = "type", .V = "integer"},
                          },
                  },
              },
      },
  });
}

[[nodiscard]] Json shorInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Shor>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "number",
                      .V =
                          llvm::json::Object{
                              {.K = "type", .V = "integer"},
                              {.K = "minimum", .V = 3},
                              {.K = "maximum", .V = ShorOptions::MAX_NUMBER},
                              {
                                  .K = "not",
                                  .V =
                                      llvm::json::Object{
                                          {
                                              .K = "multipleOf",
                                              .V = 2,
                                          },
                                      },
                              },
                          },
                  },
                  {
                      .K = "base",
                      .V =
                          llvm::json::Object{
                              {.K = "type", .V = "integer"},
                              {.K = "minimum", .V = 2},
                              {
                                  .K = "maximum",
                                  .V = ShorOptions::MAX_NUMBER - 1,
                              },
                              {.K = "default", .V = 2},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"number"}},
  });
}

[[nodiscard]] Json teleportationInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Teleportation>(
      llvm::json::Object{{.K = "properties", .V = llvm::json::Object{}}});
}

[[nodiscard]] Json wStateInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<WState>(llvm::json::Object{
      {
          .K = "properties",
          .V =
              llvm::json::Object{
                  {
                      .K = "qubits",
                      .V =
                          llvm::json::Object{
                              {
                                  .K = "maximum",
                                  .V = std::numeric_limits<int64_t>::max(),
                              },
                              {.K = "minimum", .V = 1},
                              {.K = "type", .V = "integer"},
                          },
                  },
              },
      },
      {.K = "required", .V = llvm::json::Array{"qubits"}},
  });
}

template <class Benchmark>
[[nodiscard]] ParsedBenchmark resolveInstance(Benchmark benchmark) {
  const Json manifest = manifestJSON(benchmark);
  return {
      .instance = std::move(benchmark),
      .benchmarkId = std::string(BenchmarkMetadata<Benchmark>::id),
      .caseId = manifest.getAsObject()->getString("case_id")->str(),
      .manifestJSON = mqt::json::serialize(manifest),
  };
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  llvm::Expected<BenchmarkInstance> parse##TYPE##Instance(                     \
      const Json& parameters, const std::string_view source) {                 \
    auto parsed = parse##TYPE##Parameters(parameters, source);                 \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    return std::move(*parsed);                                                 \
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

llvm::Expected<ParsedBenchmark>
parseInstanceSpecificationJSON(const std::string_view json,
                               const std::string_view source) {
  auto root = envelope(json, source, false);
  if (!root) {
    return root.takeError();
  }
  const auto id = *root->getAsObject()->getString("benchmark");
  auto parsed =
      findBenchmark(id)->parse(*root->getAsObject()->get("parameters"), source);
  if (!parsed) {
    return parsed.takeError();
  }
  return std::visit(
      [](auto& benchmark) { return resolveInstance(std::move(benchmark)); },
      *parsed);
}

llvm::Expected<std::string>
benchmarkIdFromInstanceSpecificationJSON(const std::string_view json,
                                         const std::string_view source) {
  auto root = envelope(json, source, false);
  if (!root) {
    return root.takeError();
  }
  return root->getAsObject()->getString("benchmark")->str();
}

llvm::Expected<std::string>
benchmarkIdFromManifestJSON(const std::string_view json,
                            const std::string_view source) {
  auto root = envelope(json, source, true);
  if (!root) {
    return root.takeError();
  }
  return root->getAsObject()->getString("benchmark")->str();
}

std::string listBenchmarksJSON() {
  llvm::json::Array benchmarks;
  for (const auto& entry : REGISTRY) {
    benchmarks.emplace_back(llvm::json::Object{
        {.K = "definition_version", .V = entry.definitionVersion},
        {.K = "id", .V = std::string(entry.id)},
    });
  }
  return mqt::json::serialize(llvm::json::Object{
      {.K = "benchmarks", .V = std::move(benchmarks)},
      {.K = "schema_version", .V = SCHEMA_VERSION},
  });
}

llvm::Expected<std::string>
describeBenchmarkJSON(const std::string_view benchmark) {
  if (const auto* entry = findBenchmark(benchmark)) {
    return mqt::json::serialize(entry->instanceSpecificationSchema());
  }
  return llvm::createStringError("unsupported benchmark '" +
                                 llvm::StringRef(benchmark) + "'");
}

#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)               \
  llvm::Expected<TYPE> STEM##FromInstanceSpecificationJSON(                    \
      std::string_view json, std::string_view source) {                        \
    return parseBenchmark<TYPE>(json, source, false, parse##TYPE##Parameters); \
  }                                                                            \
  std::string toInstanceSpecificationJSON(const TYPE& benchmark) {             \
    return mqt::json::serialize(instanceSpecificationJSON(benchmark));         \
  }                                                                            \
  llvm::Expected<TYPE> STEM##FromManifestJSON(std::string_view json,           \
                                              std::string_view source) {       \
    return parseBenchmark<TYPE>(json, source, true, parse##TYPE##Parameters);  \
  }                                                                            \
  std::string toManifestJSON(const TYPE& benchmark) {                          \
    return mqt::json::serialize(manifestJSON(benchmark));                      \
  }                                                                            \
  std::string caseId(const TYPE& benchmark) {                                  \
    return semanticCaseId(semanticJSON(benchmark));                            \
  }
#include "bench/BenchmarkFamilies.inc"

llvm::Expected<Counts> countsFromJSON(const std::string_view text,
                                      const std::string_view source) {
  auto root = mqt::json::parse(text, source);
  if (!root) {
    return root.takeError();
  }
  llvm::json::Path::Root diagnostic(source);
  llvm::json::ObjectMapper object(*root, diagnostic);
  uint64_t version = 0;
  if (!object || !keys(*root, {"schema_version", "counts"}, diagnostic) ||
      !object.map("schema_version", version)) {
    return diagnostic.getError();
  }
  if (version != SCHEMA_VERSION) {
    return fail(source, "$/schema_version", "must be 1");
  }
  const auto* counts = root->getAsObject()->getObject("counts");
  if ((counts == nullptr) || counts->empty()) {
    return fail(source, "$/counts", "must be a non-empty object");
  }
  Counts result;
  size_t shots = 0;
  for (const auto& [key, value] : *counts) {
    const llvm::StringRef outcome = key;
    if (outcome.empty() || !std::ranges::all_of(outcome, [](char bit) {
          return bit == '0' || bit == '1';
        })) {
      return fail(source, "$/counts", "outcomes must be non-empty bitstrings");
    }
    const auto count = value.getAsUINT64();
    if (!count || *count == 0) {
      return fail(source, "$/counts/" + outcome.str(),
                  "must be a positive integer");
    }
    if (*count > std::numeric_limits<size_t>::max() - shots) {
      return fail(source, "$/counts", "total shot count exceeds size_t");
    }
    shots += *count;
    result.emplace(outcome.str(), static_cast<size_t>(*count));
  }
  return result;
}

llvm::Expected<std::string> evaluateJSON(const std::string_view manifest,
                                         const std::string_view counts,
                                         const std::string_view manifestSource,
                                         const std::string_view countsSource) {
  auto root = envelope(manifest, manifestSource, true);
  if (!root) {
    return root.takeError();
  }
  auto parsedCounts = countsFromJSON(counts, countsSource);
  if (!parsedCounts) {
    return parsedCounts.takeError();
  }
  const auto id = *root->getAsObject()->getString("benchmark");
  auto parsed = findBenchmark(id)->parse(
      *root->getAsObject()->get("parameters"), manifestSource);
  if (!parsed) {
    return parsed.takeError();
  }
  const auto shots = std::accumulate(
      parsedCounts->begin(), parsedCounts->end(), size_t{0},
      [](size_t sum, const auto& item) { return sum + item.second; });
  try {
    return std::visit(
        [&](const auto& benchmark) -> llvm::Expected<std::string> {
          if (!matchesManifest(*root, benchmark)) {
            return fail(
                manifestSource, "$",
                "does not match its resolved benchmark instance and case ID");
          }
          return evaluationToJSON(*root->getAsObject()->getString("case_id"),
                                  shots, benchmark.evaluate(*parsedCounts));
        },
        *parsed);
  } catch (const std::invalid_argument& error) {
    return fail(countsSource, "$/counts", error.what());
  }
}

llvm::Expected<std::string> evaluationToJSON(const std::string_view caseIdValue,
                                             const size_t shots,
                                             const Evaluation& evaluation) {
  if (!validCaseId(caseIdValue)) {
    return llvm::createStringError(
        "case ID must be a full lowercase SHA-256 ID");
  }
  if (shots == 0) {
    return llvm::createStringError("evaluation requires at least one shot");
  }
  const auto validMetric = [](const double value) {
    return std::isfinite(value) && value >= 0. && value <= 1.;
  };
  if (!validMetric(evaluation.totalVariationDistance) ||
      !validMetric(evaluation.squaredHellingerFidelity) ||
      (evaluation.successProbability &&
       !validMetric(*evaluation.successProbability))) {
    return llvm::createStringError(
        "evaluation metrics must be finite and in [0, 1]");
  }

  Json success = nullptr;
  if (evaluation.successProbability) {
    success = *evaluation.successProbability;
  }
  return mqt::json::serialize(llvm::json::Object{
      {.K = "case_id", .V = std::string(caseIdValue)},
      {
          .K = "metrics",
          .V =
              llvm::json::Object{
                  {
                      .K = "squared_hellinger_fidelity",
                      .V = evaluation.squaredHellingerFidelity,
                  },
                  {.K = "success_probability", .V = std::move(success)},
                  {
                      .K = "total_variation_distance",
                      .V = evaluation.totalVariationDistance,
                  },
              },
      },
      {.K = "schema_version", .V = SCHEMA_VERSION},
      {.K = "shots", .V = shots},
  });
}

llvm::Expected<std::string> evaluationToJSON(const std::string_view caseIdValue,
                                             const size_t shots,
                                             const QAOAEvaluation& evaluation) {
  if (!validCaseId(caseIdValue) || shots == 0) {
    return llvm::createStringError(
        "evaluation requires a SHA-256 case ID and at least one shot");
  }
  if (!std::isfinite(evaluation.meanCut) || evaluation.meanCut < 0. ||
      !std::isfinite(evaluation.approximationRatio) ||
      evaluation.approximationRatio < 0. ||
      evaluation.approximationRatio > 1.) {
    return llvm::createStringError(
        "QAOA evaluation requires a finite non-negative "
        "mean cut and an approximation ratio in [0, 1]");
  }
  return mqt::json::serialize(llvm::json::Object{
      {.K = "case_id", .V = std::string(caseIdValue)},
      {
          .K = "metrics",
          .V =
              llvm::json::Object{
                  {
                      .K = "approximation_ratio",
                      .V = evaluation.approximationRatio,
                  },
                  {.K = "mean_cut", .V = evaluation.meanCut},
              },
      },
      {.K = "schema_version", .V = SCHEMA_VERSION},
      {.K = "shots", .V = shots},
  });
}

llvm::Expected<std::string> evaluationToJSON(std::string_view caseIdValue,
                                             size_t shots,
                                             const ShorEvaluation& evaluation) {
  if (!validCaseId(caseIdValue) || shots == 0) {
    return llvm::createStringError(
        "evaluation requires a SHA-256 case ID and at least one shot");
  }
  if (!std::isfinite(evaluation.successProbability) ||
      evaluation.successProbability < 0. ||
      evaluation.successProbability > 1. ||
      evaluation.factors.has_value() != (evaluation.successProbability > 0.)) {
    return llvm::createStringError("factor verification requires a success "
                                   "fraction in [0, 1] and factors on success");
  }
  Json factors = nullptr;
  if (evaluation.factors) {
    const auto [first, second] = *evaluation.factors;
    if (first < 2 || first > second ||
        second > ShorOptions::MAX_NUMBER / first) {
      return llvm::createStringError(
          "factors must form a sorted nontrivial pair "
          "within the supported range");
    }
    factors = llvm::json::Array{first, second};
  }
  return mqt::json::serialize(llvm::json::Object{
      {.K = "case_id", .V = std::string(caseIdValue)},
      {.K = "factors", .V = factors},
      {
          .K = "metrics",
          .V =
              llvm::json::Object{
                  {
                      .K = "success_probability",
                      .V = evaluation.successProbability,
                  },
              },
      },
      {.K = "schema_version", .V = SCHEMA_VERSION},
      {.K = "shots", .V = shots},
  });
}

} // namespace mqt::bench
