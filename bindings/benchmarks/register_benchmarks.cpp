/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/Evaluation.hpp"
#include "benchmarks/GHZ.hpp"
#include "benchmarks/Grover.hpp"
#include "benchmarks/JSON.hpp"
#include "benchmarks/QPE.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/optional.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

[[nodiscard]] benchmarks::Phase phaseFromPython(const nb::object& value) {
  if (nb::isinstance<benchmarks::Phase>(value)) {
    return nb::cast<const benchmarks::Phase&>(value);
  }

  const auto fraction = nb::module_::import_("fractions").attr("Fraction");
  if (!nb::isinstance(value, fraction)) {
    throw nb::type_error("phase must be a fractions.Fraction or Phase");
  }

  const auto denominator = value.attr("denominator");
  uint64_t denominatorValue;
  try {
    denominatorValue = nb::cast<uint64_t>(denominator);
  } catch (const nb::cast_error&) {
    throw nb::value_error("phase denominator must fit in 64 bits");
  }
  const auto numerator = value.attr("numerator").attr("__mod__")(denominator);
  return {nb::cast<uint64_t>(numerator), denominatorValue};
}

[[nodiscard]] nb::object asFraction(const benchmarks::Phase& phase) {
  return nb::module_::import_("fractions")
      .attr("Fraction")(phase.numerator(), phase.denominator());
}

template <class T> [[nodiscard]] nb::object generateProgram(const T& value) {
  return nb::module_::import_("mqt.core.mlir")
      .attr("_generate_benchmark")(benchmarks::toRequestJSON(value));
}

} // namespace

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = "Typed benchmark instances and analytic references.";

  nb::class_<benchmarks::Output>(m, "Output",
                                 "One logical classical output register.")
      .def_ro("name", &benchmarks::Output::name, "The register name.")
      .def_ro("width", &benchmarks::Output::width,
              "The number of big-endian outcome bits.");

  nb::class_<benchmarks::Evaluation>(
      m, "Evaluation", "The result of comparing counts with a reference.")
      .def_ro("total_variation_distance",
              &benchmarks::Evaluation::totalVariationDistance,
              "The total variation distance from the ideal distribution.")
      .def_ro("squared_hellinger_fidelity",
              &benchmarks::Evaluation::squaredHellingerFidelity,
              "The squared Hellinger fidelity with the ideal distribution.")
      .def_ro("success_probability",
              &benchmarks::Evaluation::successProbability,
              "The observed success probability, when defined.");

  nb::enum_<benchmarks::GHZTopology>(m, "GHZTopology",
                                     "Entangling topology for GHZ preparation.")
      .value("LINEAR", benchmarks::GHZTopology::Linear)
      .value("STAR", benchmarks::GHZTopology::Star);
  nb::enum_<benchmarks::GHZBasis>(m, "GHZBasis",
                                  "Measurement basis for GHZ verification.")
      .value("Z", benchmarks::GHZBasis::Z)
      .value("X", benchmarks::GHZBasis::X);

  nb::class_<benchmarks::GHZOptions>(m, "GHZOptions",
                                     "Parameters for a GHZ benchmark.")
      .def(nb::init<size_t, benchmarks::GHZTopology, benchmarks::GHZBasis>(),
           nb::kw_only(), "qubits"_a,
           "topology"_a = benchmarks::GHZTopology::Linear,
           "basis"_a = benchmarks::GHZBasis::Z)
      .def_ro("qubits", &benchmarks::GHZOptions::qubits,
              "The number of qubits.")
      .def_ro("topology", &benchmarks::GHZOptions::topology,
              "The entangling topology.")
      .def_ro("basis", &benchmarks::GHZOptions::basis,
              "The measurement basis.");

  auto ghz =
      nb::class_<benchmarks::GHZ>(m, "GHZ", "A validated GHZ benchmark.");
  ghz.def(nb::init<benchmarks::GHZOptions>(), "options"_a)
      .def_prop_ro("options", &benchmarks::GHZ::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &benchmarks::GHZ::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &benchmarks::GHZ::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &benchmarks::GHZ::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generateProgram<benchmarks::GHZ>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "request_json",
          [](const benchmarks::GHZ& value) {
            return benchmarks::toRequestJSON(value);
          },
          "The canonical request JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const benchmarks::GHZ& value) {
            return benchmarks::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const benchmarks::GHZ& value) {
            return benchmarks::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_request_json", &benchmarks::ghzFromRequestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<request>",
                  "Parse a strict benchmark request.")
      .def_static("from_manifest_json", &benchmarks::ghzFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");

  nb::class_<benchmarks::GroverOptions>(m, "GroverOptions",
                                        "Parameters for a Grover benchmark.")
      .def(nb::init<std::string, std::optional<size_t>>(), nb::kw_only(),
           "marked_bitstring"_a, "iterations"_a = nb::none())
      .def_ro("marked_bitstring", &benchmarks::GroverOptions::markedBitstring,
              "The big-endian marked outcome.")
      .def_ro("iterations", &benchmarks::GroverOptions::iterations,
              "The iteration count, or ``None`` for automatic selection.");

  auto grover = nb::class_<benchmarks::Grover>(
      m, "Grover", "A validated single-solution Grover benchmark.");
  grover.def(nb::init<benchmarks::GroverOptions>(), "options"_a)
      .def_prop_ro("options", &benchmarks::Grover::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &benchmarks::Grover::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro("qubits", &benchmarks::Grover::qubits,
                   "The number of search qubits.")
      .def("probability", &benchmarks::Grover::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &benchmarks::Grover::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generateProgram<benchmarks::Grover>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "request_json",
          [](const benchmarks::Grover& value) {
            return benchmarks::toRequestJSON(value);
          },
          "The canonical request JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const benchmarks::Grover& value) {
            return benchmarks::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const benchmarks::Grover& value) {
            return benchmarks::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_request_json", &benchmarks::groverFromRequestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<request>",
                  "Parse a strict benchmark request.")
      .def_static("from_manifest_json", &benchmarks::groverFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");

  nb::class_<benchmarks::Phase>(m, "Phase",
                                "An exact phase in turns modulo one turn.")
      .def(nb::init<uint64_t, uint64_t>(), nb::kw_only(), "numerator"_a,
           "denominator"_a)
      .def_prop_ro("numerator", &benchmarks::Phase::numerator,
                   "The reduced numerator.")
      .def_prop_ro("denominator", &benchmarks::Phase::denominator,
                   "The reduced denominator.");

  nb::enum_<benchmarks::QPEMethod>(m, "QPEMethod",
                                   "Circuit method for phase estimation.")
      .value("STANDARD", benchmarks::QPEMethod::Standard)
      .value("ITERATIVE", benchmarks::QPEMethod::Iterative);

  nb::class_<benchmarks::QPEOptions>(m, "QPEOptions",
                                     "Parameters for a QPE benchmark.")
      .def(
          "__init__",
          [](benchmarks::QPEOptions* self, const size_t precision,
             const nb::object& phase, const benchmarks::QPEMethod method) {
            new (self) benchmarks::QPEOptions{precision, phaseFromPython(phase),
                                              method};
          },
          nb::kw_only(), "precision"_a, "phase"_a,
          "method"_a = benchmarks::QPEMethod::Standard,
          nb::sig("def __init__(self, *, precision: int, phase: "
                  "fractions.Fraction | Phase, method: QPEMethod = ...) -> "
                  "None"))
      .def_ro("precision", &benchmarks::QPEOptions::precision,
              "The number of measured phase bits.")
      .def_prop_ro(
          "phase",
          [](const benchmarks::QPEOptions& options) {
            return asFraction(options.phase);
          },
          nb::sig("def phase(self) -> fractions.Fraction"),
          "The reduced phase in turns.")
      .def_ro("method", &benchmarks::QPEOptions::method, "The circuit method.");

  auto qpe =
      nb::class_<benchmarks::QPE>(m, "QPE", "A validated QPE benchmark.");
  qpe.def(nb::init<benchmarks::QPEOptions>(), "options"_a)
      .def_prop_ro("options", &benchmarks::QPE::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &benchmarks::QPE::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &benchmarks::QPE::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &benchmarks::QPE::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generateProgram<benchmarks::QPE>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "request_json",
          [](const benchmarks::QPE& value) {
            return benchmarks::toRequestJSON(value);
          },
          "The canonical request JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const benchmarks::QPE& value) {
            return benchmarks::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const benchmarks::QPE& value) {
            return benchmarks::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_request_json", &benchmarks::qpeFromRequestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<request>",
                  "Parse a strict benchmark request.")
      .def_static("from_manifest_json", &benchmarks::qpeFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
