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
#include "bench/GroverWeakMeasurement.hpp"
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
#include "bench/VQEAnsatz.hpp"
#include "bench/WState.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace mqt::bench {
namespace {

using Json = llvm::json::Value;

constexpr uint64_t SCHEMA_VERSION = 1;
constexpr std::string_view CASE_DOMAIN = "mqt-core:benchmark-case:v1";

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

template <class Benchmark>
[[nodiscard]] llvm::Expected<Benchmark>
withParameterSource(const std::string_view source,
                    llvm::Expected<Benchmark> result) {
  if (!result) {
    return fail(source, "$/parameters", llvm::toString(result.takeError()));
  }
  return result;
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

// LLVM's integer mappers do not cover size_t on every supported platform.
bool sizeValue(const Json& parameters, llvm::StringRef key, size_t& out,
               llvm::json::Path path, bool optional = false) {
  const auto* field = parameters.getAsObject()->get(key);
  if (field == nullptr) {
    if (optional) {
      return true;
    }
    path.field(key).report("missing value");
    return false;
  }
  const auto value = field->getAsUINT64();
  if (!value || !std::in_range<size_t>(*value)) {
    path.field(key).report("expected uint64_t fitting size_t");
    return false;
  }
  out = static_cast<size_t>(*value);
  return true;
}

[[nodiscard]] llvm::Expected<Json> envelope(const std::string_view text,
                                            const std::string_view source,
                                            const bool manifest) {
  auto root = llvm::json::parse(text);
  if (!root) {
    return llvm::createStringError(llvm::StringRef(source) + ": " +
                                   llvm::toString(root.takeError()));
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
  if (version != SCHEMA_VERSION) {
    return fail(source, "$/schema_version", "must be 1");
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
  return withParameterSource(source, BV::create(std::move(options)));
}

[[nodiscard]] llvm::Expected<FanOut>
parseFanOutParameters(const Json& parameters, std::string_view source) {
  FanOutOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"controls"}, path) ||
      !sizeValue(parameters, "controls", options.controls, path)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, FanOut::create(options));
}

[[nodiscard]] llvm::Expected<GHZ> parseGHZParameters(const Json& parameters,
                                                     std::string_view source) {
  GHZOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"qubits", "topology", "basis"}, path) ||
      !sizeValue(parameters, "qubits", options.qubits, path) ||
      !enumValue<GHZTopology>(
          parameters, "topology", options.topology,
          {{"linear", GHZTopology::Linear}, {"star", GHZTopology::Star}},
          path) ||
      !enumValue<GHZBasis>(parameters, "basis", options.basis,
                           {{"z", GHZBasis::Z}, {"x", GHZBasis::X}}, path)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, GHZ::create(options));
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
      !sizeValue(parameters, "iterations", iterations, path, true)) {
    return diagnostic.getError();
  }
  if (parameters.getAsObject()->get("iterations") != nullptr) {
    options.iterations = iterations;
  }
  return withParameterSource(source, Grover::create(std::move(options)));
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
  return withParameterSource(source,
                             GroverWeakMeasurement::create(std::move(options)));
}

[[nodiscard]] llvm::Expected<MagicStateDistillation>
parseMagicStateDistillationParameters(const Json& parameters,
                                      std::string_view source) {
  MagicStateDistillationOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"levels"}, path) ||
      !sizeValue(parameters, "levels", options.levels, path, true)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, MagicStateDistillation::create(options));
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
  return withParameterSource(source,
                             ModularMultiplier::create(std::move(options)));
}

[[nodiscard]] llvm::Expected<Multiplexer>
parseMultiplexerParameters(const Json& parameters, std::string_view source) {
  MultiplexerOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"qubits"}, path) ||
      !sizeValue(parameters, "qubits", options.qubits, path)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, Multiplexer::create(options));
}

[[nodiscard]] llvm::Expected<QAOA>
parseQAOAParameters(const Json& parameters, std::string_view source) {
  QAOAOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"qubits", "depth"}, path) ||
      !sizeValue(parameters, "qubits", options.qubits, path) ||
      !sizeValue(parameters, "depth", options.depth, path, true)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, QAOA::create(options));
}

[[nodiscard]] llvm::Expected<QFT> parseQFTParameters(const Json& parameters,
                                                     std::string_view source) {
  QFTOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"qubits", "period_exponent", "method"}, path) ||
      !sizeValue(parameters, "qubits", options.qubits, path) ||
      !sizeValue(parameters, "period_exponent", options.periodExponent, path) ||
      !enumValue<QFTMethod>(parameters, "method", options.method,
                            {
                                {"standard", QFTMethod::Standard},
                                {"semiclassical", QFTMethod::Semiclassical},
                            },
                            path)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, QFT::create(options));
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
  return withParameterSource(source, QFTAdder::create(std::move(options)));
}

[[nodiscard]] llvm::Expected<RepeatUntilSuccess>
parseRepeatUntilSuccessParameters(const Json& parameters,
                                  std::string_view source) {
  RepeatUntilSuccessOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"data_qubits"}, path) ||
      !sizeValue(parameters, "data_qubits", options.dataQubits, path, true)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, RepeatUntilSuccess::create(options));
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
  return withParameterSource(source, Shor::create(options));
}

[[nodiscard]] llvm::Expected<WState>
parseWStateParameters(const Json& parameters, std::string_view source) {
  WStateOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"qubits"}, path) ||
      !sizeValue(parameters, "qubits", options.qubits, path)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, WState::create(options));
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
  size_t precision = 0;
  auto method = QPEMethod::Standard;
  if (!keys(parameters, {"precision", "phase", "method"}, path) ||
      !sizeValue(parameters, "precision", precision, path) ||
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
  auto parsedPhase = Phase::create(numerator, denominator);
  if (!parsedPhase) {
    return fail(source, "$/parameters",
                llvm::toString(parsedPhase.takeError()));
  }
  return withParameterSource(source, QPE::create({
                                         .precision = precision,
                                         .phase = *parsedPhase,
                                         .method = method,
                                     }));
}

[[nodiscard]] llvm::Expected<VQEAnsatz>
parseVQEAnsatzParameters(const Json& parameters, std::string_view source) {
  VQEAnsatzOptions options{};
  const auto diagnosticSource = std::string(source) + ":$/parameters";
  llvm::json::Path::Root diagnostic(diagnosticSource);
  const llvm::json::Path path(diagnostic);
  if (!keys(parameters, {"qubits", "depth"}, path) ||
      !sizeValue(parameters, "qubits", options.qubits, path) ||
      !sizeValue(parameters, "depth", options.depth, path, true)) {
    return diagnostic.getError();
  }
  return withParameterSource(source, VQEAnsatz::create(options));
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
  return llvm::json::Object{{"number", options.number}, {"base", options.base}};
}

[[nodiscard]] Json parametersJSON(const BV& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"hidden_bitstring", options.hiddenBitstring},
      {"method", methodName(options.method)},
  };
}

[[nodiscard]] Json parametersJSON(const FanOut& benchmark) {
  return llvm::json::Object{{"controls", benchmark.options().controls}};
}

[[nodiscard]] Json parametersJSON(const GHZ& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"basis", basisName(options.basis)},
      {"qubits", options.qubits},
      {"topology", topologyName(options.topology)},
  };
}

[[nodiscard]] Json parametersJSON(const Grover& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"iterations", *options.iterations},
      {"marked_bitstring", options.markedBitstring},
  };
}

[[nodiscard]] Json parametersJSON(const GroverWeakMeasurement& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"marked_bitstring", options.markedBitstring},
      {"measurement_strength", *options.measurementStrength},
  };
}

[[nodiscard]] Json parametersJSON(const MagicStateDistillation& benchmark) {
  return llvm::json::Object{{"levels", benchmark.options().levels}};
}

[[nodiscard]] Json parametersJSON(const ModularMultiplier& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"control", std::string(1, options.control)},
      {"multiplicand", options.multiplicand},
      {"modulus", options.modulus},
      {"multiplier", options.multiplier},
  };
}

[[nodiscard]] Json parametersJSON(const Multiplexer& benchmark) {
  return llvm::json::Object{{"qubits", benchmark.options().qubits}};
}

[[nodiscard]] Json parametersJSON(const QAOA& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"depth", options.depth},
      {"qubits", options.qubits},
  };
}

[[nodiscard]] Json parametersJSON(const QFT& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"method", methodName(options.method)},
      {"period_exponent", options.periodExponent},
      {"qubits", options.qubits},
  };
}

[[nodiscard]] Json parametersJSON(const QFTAdder& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
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
  return llvm::json::Object{
      {"method", methodName(options.method)},
      {
          "phase",
          llvm::json::Object{
              {"denominator", options.phase.denominator()},
              {"numerator", options.phase.numerator()},
          },
      },
      {"precision", options.precision},
  };
}

[[nodiscard]] Json parametersJSON(const RepeatUntilSuccess& benchmark) {
  return llvm::json::Object{{"data_qubits", benchmark.options().dataQubits}};
}

[[nodiscard]] Json parametersJSON(const Teleportation& /*unused*/) {
  return llvm::json::Object{};
}

[[nodiscard]] Json parametersJSON(const VQEAnsatz& benchmark) {
  const auto& options = benchmark.options();
  return llvm::json::Object{
      {"depth", options.depth},
      {"qubits", options.qubits},
  };
}

[[nodiscard]] Json parametersJSON(const WState& benchmark) {
  return llvm::json::Object{{"qubits", benchmark.options().qubits}};
}

[[nodiscard]] Json analyticReferenceJSON(
    const Output& output, const std::string_view model,
    const std::optional<std::string_view> successOutcome = std::nullopt) {
  llvm::json::Object reference = llvm::json::Object{
      {"kind", "analytic"},
      {"model", std::string(model)},
      {"outcome_order", "big_endian"},
      {"output", output.name},
      {"version", uint64_t{1}},
  };
  if (successOutcome) {
    reference["success_outcome"] = std::string(*successOutcome);
  }
  return reference;
}

[[nodiscard]] Json referenceJSON(const QAOA& benchmark) {
  return llvm::json::Object{
      {"kind", "verification"},        {"model", "qaoa_max_cut_cycle"},
      {"outcome_order", "big_endian"}, {"output", benchmark.output().name},
      {"version", uint64_t{1}},
  };
}

[[nodiscard]] Json referenceJSON(const Shor& benchmark) {
  return llvm::json::Object{
      {"kind", "verification"},        {"model", "shor_factors"},
      {"outcome_order", "big_endian"}, {"output", benchmark.output().name},
      {"version", uint64_t{1}},
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

[[nodiscard]] Json referenceJSON(const VQEAnsatz& /*unused*/) {
  return llvm::json::Object{{"kind", "none"}};
}

[[nodiscard]] Json referenceJSON(const WState& benchmark) {
  return analyticReferenceJSON(benchmark.output(), "w_state");
}

[[nodiscard]] Json semanticJSON(const std::string_view id,
                                const uint64_t definitionVersionValue,
                                const Json& parameters, const Output& output,
                                const Json& reference) {
  return llvm::json::Object{
      {"benchmark", std::string(id)},
      {"definition_version", definitionVersionValue},
      {
          "outputs",
          llvm::json::Array{
              llvm::json::Object{
                  {"name", output.name},
                  {"width", output.width},
              },
          },
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
  llvm::raw_string_ostream raw(input);
  raw << semantic;
  return "sha256-" +
         llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(input)),
                     true);
}

template <class Benchmark>
[[nodiscard]] Json manifestJSON(const Benchmark& benchmark) {
  auto semantic = semanticJSON(benchmark);
  (*semantic.getAsObject())["case_id"] = semanticCaseId(semantic);
  (*semantic.getAsObject())["schema_version"] = SCHEMA_VERSION;
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
  if (manifest && *root != manifestJSON(*benchmark)) {
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
                  "benchmark",
                  llvm::json::Object{{"const", std::string(Metadata::id)}},
              },
              {"parameters", std::move(parameters)},
              {"schema_version", llvm::json::Object{{"const", SCHEMA_VERSION}}},
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
          "properties",
          llvm::json::Object{
              {
                  "hidden_bitstring",
                  llvm::json::Object{
                      {"maxLength", BVOptions::MAX_BITS},
                      {"minLength", 1},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
              {
                  "method",
                  llvm::json::Object{
                      {"default", "static"},
                      {"enum", llvm::json::Array{"static", "dynamic"}},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"hidden_bitstring"}},
  });
}

[[nodiscard]] Json fanOutInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<FanOut>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "controls",
                  llvm::json::Object{
                      {"maximum", FanOutOptions::MAX_CONTROLS},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"controls"}},
  });
}

[[nodiscard]] Json ghzInstanceSpecificationSchema() {
  auto parameters = llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "basis",
                  llvm::json::Object{
                      {"default", "z"},
                      {"enum", llvm::json::Array{"z", "x"}},
                  },
              },
              {
                  "qubits",
                  llvm::json::Object{
                      {"maximum", GHZOptions::MAX_QUBITS},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
              {
                  "topology",
                  llvm::json::Object{
                      {"default", "linear"},
                      {"enum", llvm::json::Array{"linear", "star"}},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"qubits"}},
  };
  parameters["allOf"] = llvm::json::Array{
      llvm::json::Object{
          {
              "if",
              llvm::json::Object{
                  {
                      "properties",
                      llvm::json::Object{
                          {"basis", llvm::json::Object{{"const", "x"}}},
                      },
                  },
                  {"required", llvm::json::Array{"basis"}},
              },
          },
          {
              "then",
              llvm::json::Object{
                  {
                      "properties",
                      llvm::json::Object{
                          {
                              "qubits",
                              llvm::json::Object{
                                  {
                                      "maximum",
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
          "properties",
          llvm::json::Object{
              {
                  "iterations",
                  llvm::json::Object{
                      {"maximum", std::numeric_limits<int32_t>::max()},
                      {"minimum", 0},
                      {"type", "integer"},
                  },
              },
              {
                  "marked_bitstring",
                  llvm::json::Object{
                      {"maxLength", 62},
                      {"minLength", 2},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"marked_bitstring"}},
  });
}

[[nodiscard]] Json groverWeakMeasurementInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<GroverWeakMeasurement>(
      llvm::json::Object{
          {
              "properties",
              llvm::json::Object{
                  {
                      "marked_bitstring",
                      llvm::json::Object{
                          {
                              "maxLength",
                              GroverWeakMeasurementOptions::MAX_QUBITS,
                          },
                          {"minLength", 2},
                          {"pattern", "^[01]+$"},
                          {"type", "string"},
                      },
                  },
                  {
                      "measurement_strength",
                      llvm::json::Object{
                          {"exclusiveMinimum", 0},
                          {"maximum", 0.5},
                          {"type", "number"},
                      },
                  },
              },
          },
          {"required", llvm::json::Array{"marked_bitstring"}},
      });
}

[[nodiscard]] Json magicStateDistillationInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<MagicStateDistillation>(
      llvm::json::Object{
          {
              "properties",
              llvm::json::Object{
                  {
                      "levels",
                      llvm::json::Object{
                          {"default", 1},
                          {"minimum", 1},
                          {"maximum", std::numeric_limits<int64_t>::max() / 5},
                          {"type", "integer"},
                      },
                  },
              },
          },
      });
}

[[nodiscard]] Json modularMultiplierInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<ModularMultiplier>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "control",
                  llvm::json::Object{
                      {"default", "1"},
                      {"enum", llvm::json::Array{"0", "1", "+"}},
                  },
              },
              {
                  "multiplicand",
                  llvm::json::Object{
                      {"type", "string"},
                      {"minLength", 2},
                      {"maxLength", ModularMultiplierOptions::MAX_BITS},
                      {"pattern", "^[01+]+$"},
                  },
              },
              {
                  "modulus",
                  llvm::json::Object{
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
                  llvm::json::Object{
                      {"maxLength", ModularMultiplierOptions::MAX_BITS},
                      {"minLength", 2},
                      {"pattern", "^[01]+$"},
                      {"type", "string"},
                  },
              },
          },
      },
      {
          "required",
          llvm::json::Array{"multiplier", "modulus", "multiplicand"},
      },
  });
}

[[nodiscard]] Json multiplexerInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Multiplexer>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "qubits",
                  llvm::json::Object{
                      {"maximum", MultiplexerOptions::MAX_QUBITS},
                      {"minimum", 2},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"qubits"}},
  });
}

[[nodiscard]] Json qaoaInstanceSpecificationSchema() {
  const QAOAOptions defaults{.qubits = 3};
  return baseInstanceSpecificationSchema<QAOA>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "depth",
                  llvm::json::Object{
                      {"default", defaults.depth},
                      {"maximum", QAOAOptions::MAX_DEPTH},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
              {
                  "qubits",
                  llvm::json::Object{
                      {"maximum", std::numeric_limits<int64_t>::max()},
                      {"minimum", 3},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"qubits"}},
  });
}

[[nodiscard]] Json qftInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<QFT>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "method",
                  llvm::json::Object{
                      {"default", "standard"},
                      {"enum", llvm::json::Array{"standard", "semiclassical"}},
                  },
              },
              {
                  "period_exponent",
                  llvm::json::Object{
                      {"maximum", QFTOptions::MAX_PERIOD_EXPONENT},
                      {"minimum", 0},
                      {"type", "integer"},
                  },
              },
              {
                  "qubits",
                  llvm::json::Object{
                      {"maximum", QFTOptions::MAX_QUBITS},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"qubits", "period_exponent"}},
  });
}

[[nodiscard]] Json qftAdderInstanceSpecificationSchema() {
  llvm::json::Object properties{
      {
          "addend",
          llvm::json::Object{
              {"type", "string"},
              {"minLength", 1},
              {"maxLength", QFTAdderOptions::MAX_QUBITS},
              {"pattern", "^[01+]+$"},
          },
      },
      {
          "accumulator",
          llvm::json::Object{
              {"type", "string"},
              {"minLength", 1},
              {"maxLength", QFTAdderOptions::MAX_QUBITS},
              {"pattern", "^[01]+$"},
          },
      },
      {
          "method",
          llvm::json::Object{
              {"type", "string"},
              {"enum", llvm::json::Array{"register", "constant"}},
              {"default", "register"},
          },
      },
      {
          "overflow",
          llvm::json::Object{
              {"type", "string"},
              {"enum", llvm::json::Array{"wrap", "carry"}},
              {"default", "wrap"},
          },
      },
  };
  llvm::json::Object constantCondition{
      {
          "properties",
          llvm::json::Object{
              {"method", llvm::json::Object{{"const", "constant"}}},
          },
      },
      {"required", llvm::json::Array{"method"}},
  };
  llvm::json::Object constantProperties{
      {"addend", llvm::json::Object{{"pattern", "^[01]+$"}}},
  };
  llvm::json::Object carryCondition{
      {
          "properties",
          llvm::json::Object{
              {"overflow", llvm::json::Object{{"const", "carry"}}},
          },
      },
      {"required", llvm::json::Array{"overflow"}},
  };
  llvm::json::Object carryProperties{
      {
          "addend",
          llvm::json::Object{{"maxLength", QFTAdderOptions::MAX_QUBITS - 1U}},
      },
      {
          "accumulator",
          llvm::json::Object{{"maxLength", QFTAdderOptions::MAX_QUBITS - 1U}},
      },
  };
  llvm::json::Array constraints{
      llvm::json::Object{
          {"if", std::move(constantCondition)},
          {
              "then",
              llvm::json::Object{{"properties", std::move(constantProperties)}},
          },
      },
      llvm::json::Object{
          {"if", std::move(carryCondition)},
          {
              "then",
              llvm::json::Object{{"properties", std::move(carryProperties)}},
          },
      },
  };
  return baseInstanceSpecificationSchema<QFTAdder>(llvm::json::Object{
      {"properties", std::move(properties)},
      {"allOf", std::move(constraints)},
      {"required", llvm::json::Array{"addend", "accumulator"}},
  });
}

[[nodiscard]] Json qpeInstanceSpecificationSchema() {
  llvm::json::Object phaseProperties{
      {
          "denominator",
          llvm::json::Object{
              {"maximum", std::numeric_limits<uint64_t>::max()},
              {"minimum", 1},
              {"type", "integer"},
          },
      },
      {
          "numerator",
          llvm::json::Object{
              {"maximum", std::numeric_limits<uint64_t>::max()},
              {"minimum", 0},
              {"type", "integer"},
          },
      },
  };
  llvm::json::Object phase{
      {"additionalProperties", false},
      {"properties", std::move(phaseProperties)},
      {"required", llvm::json::Array{"numerator", "denominator"}},
      {"type", "object"},
  };
  llvm::json::Object properties{
      {
          "method",
          llvm::json::Object{
              {"default", "standard"},
              {
                  "enum",
                  llvm::json::Array{
                      "standard",
                      "iterative",
                  },
              },
          },
      },
      {"phase", std::move(phase)},
      {
          "precision",
          llvm::json::Object{
              {"maximum", QPEOptions::MAX_PRECISION},
              {"minimum", 1},
              {"type", "integer"},
          },
      },
  };
  return baseInstanceSpecificationSchema<QPE>(llvm::json::Object{
      {"properties", std::move(properties)},
      {"required", llvm::json::Array{"precision", "phase"}},
  });
}

[[nodiscard]] Json repeatUntilSuccessInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<RepeatUntilSuccess>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "data_qubits",
                  llvm::json::Object{
                      {"default", 1},
                      {"minimum", 1},
                      {"maximum", RepeatUntilSuccessOptions::MAX_DATA_QUBITS},
                      {"type", "integer"},
                  },
              },
          },
      },
  });
}

[[nodiscard]] Json shorInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Shor>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "number",
                  llvm::json::Object{
                      {"type", "integer"},
                      {"minimum", 3},
                      {"maximum", ShorOptions::MAX_NUMBER},
                      {"not", llvm::json::Object{{"multipleOf", 2}}},
                  },
              },
              {
                  "base",
                  llvm::json::Object{
                      {"type", "integer"},
                      {"minimum", 2},
                      {"maximum", ShorOptions::MAX_NUMBER - 1},
                      {"default", 2},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"number"}},
  });
}

[[nodiscard]] Json teleportationInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<Teleportation>(
      llvm::json::Object{{"properties", llvm::json::Object{}}});
}

[[nodiscard]] Json vqeAnsatzInstanceSpecificationSchema() {
  const VQEAnsatzOptions defaults{.qubits = 1};
  return baseInstanceSpecificationSchema<VQEAnsatz>(llvm::json::Object{
      {"description", "The product qubits * (depth + 1) must fit a signed "
                      "64-bit circuit input dimension."},
      {
          "properties",
          llvm::json::Object{
              {
                  "depth",
                  llvm::json::Object{
                      {"default", defaults.depth},
                      {"maximum", std::numeric_limits<int64_t>::max() - 1},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
              {
                  "qubits",
                  llvm::json::Object{
                      {"maximum", std::numeric_limits<int64_t>::max() / 2},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"qubits"}},
  });
}

[[nodiscard]] Json wStateInstanceSpecificationSchema() {
  return baseInstanceSpecificationSchema<WState>(llvm::json::Object{
      {
          "properties",
          llvm::json::Object{
              {
                  "qubits",
                  llvm::json::Object{
                      {"maximum", std::numeric_limits<int64_t>::max()},
                      {"minimum", 1},
                      {"type", "integer"},
                  },
              },
          },
      },
      {"required", llvm::json::Array{"qubits"}},
  });
}

template <class Benchmark>
[[nodiscard]] ParsedBenchmark resolveInstance(Benchmark benchmark) {
  const Json manifest = manifestJSON(benchmark);
  return {
      .instance = std::move(benchmark),
      .benchmarkId = std::string(BenchmarkMetadata<Benchmark>::id),
      .caseId = manifest.getAsObject()->getString("case_id")->str(),
      .manifestJSON = llvm::formatv("{0}", manifest).str(),
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

[[nodiscard]] std::string evaluationToJSON(const std::string_view caseIdValue,
                                           const Evaluation& evaluation) {
  Json success = nullptr;
  if (evaluation.successProbability) {
    success = *evaluation.successProbability;
  }
  return llvm::formatv("{0}",
                       llvm::json::Value(llvm::json::Object{
                           {"case_id", std::string(caseIdValue)},
                           {
                               "metrics",
                               llvm::json::Object{
                                   {
                                       "squared_hellinger_fidelity",
                                       evaluation.squaredHellingerFidelity,
                                   },
                                   {"success_probability", std::move(success)},
                                   {
                                       "total_variation_distance",
                                       evaluation.totalVariationDistance,
                                   },
                               },
                           },
                           {"schema_version", SCHEMA_VERSION},
                           {"shots", evaluation.shots},
                       }))
      .str();
}

[[nodiscard]] std::string evaluationToJSON(const std::string_view caseIdValue,
                                           const QAOAEvaluation& evaluation) {
  return llvm::formatv("{0}", llvm::json::Value(llvm::json::Object{
                                  {"case_id", std::string(caseIdValue)},
                                  {
                                      "metrics",
                                      llvm::json::Object{
                                          {
                                              "approximation_ratio",
                                              evaluation.approximationRatio,
                                          },
                                          {"mean_cut", evaluation.meanCut},
                                      },
                                  },
                                  {"schema_version", SCHEMA_VERSION},
                                  {"shots", evaluation.shots},
                              }))
      .str();
}

[[nodiscard]] std::string evaluationToJSON(std::string_view caseIdValue,
                                           const ShorEvaluation& evaluation) {
  Json factors = nullptr;
  if (evaluation.factors) {
    const auto [first, second] = *evaluation.factors;
    factors = llvm::json::Array{first, second};
  }
  return llvm::formatv("{0}", llvm::json::Value(llvm::json::Object{
                                  {"case_id", std::string(caseIdValue)},
                                  {"factors", factors},
                                  {
                                      "metrics",
                                      llvm::json::Object{
                                          {
                                              "success_probability",
                                              evaluation.successProbability,
                                          },
                                      },
                                  },
                                  {"schema_version", SCHEMA_VERSION},
                                  {"shots", evaluation.shots},
                              }))
      .str();
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
        {"definition_version", entry.definitionVersion},
        {"id", std::string(entry.id)},
    });
  }
  return llvm::formatv("{0}", llvm::json::Value(llvm::json::Object{
                                  {"benchmarks", std::move(benchmarks)},
                                  {"schema_version", SCHEMA_VERSION},
                              }))
      .str();
}

llvm::Expected<std::string>
describeBenchmarkJSON(const std::string_view benchmark) {
  if (const auto* entry = findBenchmark(benchmark)) {
    return llvm::formatv("{0}", entry->instanceSpecificationSchema()).str();
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
    return llvm::formatv("{0}", instanceSpecificationJSON(benchmark)).str();   \
  }                                                                            \
  llvm::Expected<TYPE> STEM##FromManifestJSON(std::string_view json,           \
                                              std::string_view source) {       \
    return parseBenchmark<TYPE>(json, source, true, parse##TYPE##Parameters);  \
  }                                                                            \
  std::string toManifestJSON(const TYPE& benchmark) {                          \
    return llvm::formatv("{0}", manifestJSON(benchmark)).str();                \
  }                                                                            \
  std::string caseId(const TYPE& benchmark) {                                  \
    return semanticCaseId(semanticJSON(benchmark));                            \
  }
#include "bench/BenchmarkFamilies.inc"

llvm::Expected<Counts> countsFromJSON(const std::string_view text,
                                      const std::string_view source) {
  auto root = llvm::json::parse(text);
  if (!root) {
    return llvm::createStringError(llvm::StringRef(source) + ": " +
                                   llvm::toString(root.takeError()));
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
  return std::visit(
      [&](const auto& benchmark) -> llvm::Expected<std::string> {
        if (*root != manifestJSON(benchmark)) {
          return fail(
              manifestSource, "$",
              "does not match its resolved benchmark instance and case ID");
        }
        if constexpr (requires { benchmark.evaluate(*parsedCounts); }) {
          auto evaluation = benchmark.evaluate(*parsedCounts);
          if (!evaluation) {
            return fail(countsSource, "$/counts",
                        llvm::toString(evaluation.takeError()));
          }
          return evaluationToJSON(*root->getAsObject()->getString("case_id"),
                                  *evaluation);
        } else {
          return fail(manifestSource, "$/reference",
                      "benchmark '" + id.str() +
                          "' does not define a built-in evaluation");
        }
      },
      *parsed);
}

} // namespace mqt::bench
