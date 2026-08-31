/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/Evaluation.hpp"
#include "bench/GHZ.hpp"
#include "bench/Grover.hpp"
#include "bench/JSON.hpp"
#include "bench/QFT.hpp"
#include "bench/QPE.hpp"

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

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

[[nodiscard]] bench::Phase phaseFromPython(const nb::object& value) {
  if (nb::isinstance<bench::Phase>(value)) {
    return nb::cast<const bench::Phase&>(value);
  }

  const auto fraction = nb::module_::import_("fractions").attr("Fraction");
  if (!nb::isinstance(value, fraction)) {
    throw nb::type_error("phase must be a fractions.Fraction or Phase");
  }

  const auto denominator = value.attr("denominator");
  uint64_t denominatorValue = 0;
  try {
    denominatorValue = nb::cast<uint64_t>(denominator);
  } catch (const nb::cast_error&) {
    throw nb::value_error("phase denominator must fit in 64 bits");
  }
  const auto numerator = value.attr("numerator").attr("__mod__")(denominator);
  return {nb::cast<uint64_t>(numerator), denominatorValue};
}

[[nodiscard]] nb::object asFraction(const bench::Phase& phase) {
  return nb::module_::import_("fractions")
      .attr("Fraction")(phase.numerator(), phase.denominator());
}

template <class T> [[nodiscard]] nb::object generate(const T& value) {
  return nb::module_::import_("mqt.core.mlir")
      .attr("_generate_benchmark")(bench::toInstanceSpecificationJSON(value));
}

} // namespace

// The nanobind module macro requires its module handle by value.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = "Typed benchmark instances and analytic references.";

  nb::class_<bench::Output>(m, "Output",
                            "One logical classical output register.")
      .def_ro("name", &bench::Output::name, "The register name.")
      .def_ro("width", &bench::Output::width,
              "The number of big-endian outcome bits.");

  nb::class_<bench::Evaluation>(
      m, "Evaluation", "The result of comparing counts with a reference.")
      .def_ro("total_variation_distance",
              &bench::Evaluation::totalVariationDistance,
              "The total variation distance from the ideal distribution.")
      .def_ro("squared_hellinger_fidelity",
              &bench::Evaluation::squaredHellingerFidelity,
              "The squared Hellinger fidelity with the ideal distribution.")
      .def_ro("success_probability", &bench::Evaluation::successProbability,
              "The observed success probability, when defined.");

  nb::enum_<bench::BVMethod>(
      m, "BVMethod",
      "Static allocation or dynamic measurement and qubit reuse.")
      .value("STATIC", bench::BVMethod::Static)
      .value("DYNAMIC", bench::BVMethod::Dynamic);

  nb::class_<bench::BVOptions>(
      m, "BVOptions", "Parameters for a Bernstein--Vazirani benchmark.")
      .def(nb::init<std::string, bench::BVMethod>(), nb::kw_only(),
           "hidden_bitstring"_a, "method"_a = bench::BVMethod::Static)
      .def_ro("hidden_bitstring", &bench::BVOptions::hiddenBitstring,
              "The big-endian hidden bitstring.")
      .def_ro("method", &bench::BVOptions::method, "The circuit method.");

  auto bv = nb::class_<bench::BV>(m, "BV",
                                  "A validated Bernstein--Vazirani benchmark.");
  bv.def(nb::init<bench::BVOptions>(), "options"_a)
      .def_prop_ro("options", &bench::BV::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::BV::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::BV::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::BV::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generate<bench::BV>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::BV& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::BV& value) { return bench::toManifestJSON(value); },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::BV& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::bvFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::bvFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");

  nb::enum_<bench::GHZTopology>(m, "GHZTopology",
                                "Entangling topology for GHZ preparation.")
      .value("LINEAR", bench::GHZTopology::Linear)
      .value("STAR", bench::GHZTopology::Star);
  nb::enum_<bench::GHZBasis>(m, "GHZBasis",
                             "Measurement basis for GHZ verification.")
      .value("Z", bench::GHZBasis::Z)
      .value("X", bench::GHZBasis::X);

  nb::class_<bench::GHZOptions>(m, "GHZOptions",
                                "Parameters for a GHZ benchmark.")
      .def(nb::init<size_t, bench::GHZTopology, bench::GHZBasis>(),
           nb::kw_only(), "qubits"_a, "topology"_a = bench::GHZTopology::Linear,
           "basis"_a = bench::GHZBasis::Z)
      .def_ro("qubits", &bench::GHZOptions::qubits, "The number of qubits.")
      .def_ro("topology", &bench::GHZOptions::topology,
              "The entangling topology.")
      .def_ro("basis", &bench::GHZOptions::basis, "The measurement basis.");

  auto ghz = nb::class_<bench::GHZ>(m, "GHZ", "A validated GHZ benchmark.");
  ghz.def(nb::init<bench::GHZOptions>(), "options"_a)
      .def_prop_ro("options", &bench::GHZ::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::GHZ::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::GHZ::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::GHZ::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generate<bench::GHZ>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::GHZ& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::GHZ& value) { return bench::toManifestJSON(value); },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::GHZ& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::ghzFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::ghzFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");

  nb::class_<bench::GroverOptions>(m, "GroverOptions",
                                   "Parameters for a Grover benchmark.")
      .def(nb::init<std::string, std::optional<size_t>>(), nb::kw_only(),
           "marked_bitstring"_a, "iterations"_a = nb::none())
      .def_ro("marked_bitstring", &bench::GroverOptions::markedBitstring,
              "The big-endian marked outcome.")
      .def_ro("iterations", &bench::GroverOptions::iterations,
              "The iteration count, or ``None`` for automatic selection.");

  auto grover = nb::class_<bench::Grover>(
      m, "Grover", "A validated single-solution Grover benchmark.");
  grover.def(nb::init<bench::GroverOptions>(), "options"_a)
      .def_prop_ro("options", &bench::Grover::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::Grover::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro("qubits", &bench::Grover::qubits,
                   "The number of search qubits.")
      .def("probability", &bench::Grover::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::Grover::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generate<bench::Grover>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::Grover& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::Grover& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::Grover& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::groverFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::groverFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");

  nb::enum_<bench::QFTMethod>(
      m, "QFTMethod",
      "Full-register or semiclassical measurement-and-feed-forward method.")
      .value("STANDARD", bench::QFTMethod::Standard)
      .value("SEMICLASSICAL", bench::QFTMethod::Semiclassical);

  nb::class_<bench::QFTOptions>(m, "QFTOptions",
                                "Parameters for a QFT benchmark.")
      .def(nb::init<size_t, size_t, bench::QFTMethod>(), nb::kw_only(),
           "qubits"_a, "period_exponent"_a,
           "method"_a = bench::QFTMethod::Standard)
      .def_ro("qubits", &bench::QFTOptions::qubits,
              "The number of transformed qubits.")
      .def_ro("period_exponent", &bench::QFTOptions::periodExponent,
              "The base-two input-period exponent.")
      .def_ro("method", &bench::QFTOptions::method, "The circuit method.");

  auto qft = nb::class_<bench::QFT>(m, "QFT", "A validated QFT benchmark.");
  qft.def(nb::init<bench::QFTOptions>(), "options"_a)
      .def_prop_ro("options", &bench::QFT::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QFT::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::QFT::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::QFT::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generate<bench::QFT>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::QFT& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::QFT& value) { return bench::toManifestJSON(value); },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::QFT& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::qftFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::qftFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");

  nb::class_<bench::Phase>(m, "Phase",
                           "An exact phase in turns modulo one turn.")
      .def(nb::init<uint64_t, uint64_t>(), nb::kw_only(), "numerator"_a,
           "denominator"_a)
      .def_prop_ro("numerator", &bench::Phase::numerator,
                   "The reduced numerator.")
      .def_prop_ro("denominator", &bench::Phase::denominator,
                   "The reduced denominator.");

  nb::enum_<bench::QPEMethod>(
      m, "QPEMethod",
      "Full-register or iterative measurement-and-feed-forward method.")
      .value("STANDARD", bench::QPEMethod::Standard)
      .value("ITERATIVE", bench::QPEMethod::Iterative);

  nb::class_<bench::QPEOptions>(m, "QPEOptions",
                                "Parameters for a QPE benchmark.")
      .def(
          "__init__",
          [](bench::QPEOptions* self, const size_t precision,
             const nb::object& phase, const bench::QPEMethod method) {
            new (self) bench::QPEOptions{
                .precision = precision,
                .phase = phaseFromPython(phase),
                .method = method,
            };
          },
          nb::kw_only(), "precision"_a, "phase"_a,
          "method"_a = bench::QPEMethod::Standard,
          nb::sig("def __init__(self, *, precision: int, phase: "
                  "fractions.Fraction | Phase, method: QPEMethod = ...) -> "
                  "None"))
      .def_ro("precision", &bench::QPEOptions::precision,
              "The number of measured phase bits.")
      .def_prop_ro(
          "phase",
          [](const bench::QPEOptions& options) {
            return asFraction(options.phase);
          },
          nb::sig("def phase(self) -> fractions.Fraction"),
          "The reduced phase in turns.")
      .def_ro("method", &bench::QPEOptions::method, "The circuit method.");

  auto qpe = nb::class_<bench::QPE>(m, "QPE", "A validated QPE benchmark.");
  qpe.def(nb::init<bench::QPEOptions>(), "options"_a)
      .def_prop_ro("options", &bench::QPE::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QPE::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::QPE::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::QPE::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def("generate", &generate<bench::QPE>,
           nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
           "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::QPE& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::QPE& value) { return bench::toManifestJSON(value); },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::QPE& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::qpeFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::qpeFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
