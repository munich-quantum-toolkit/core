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
#include "bench/QPE.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <cstdint>
#include <new>

namespace mqt::bindings {

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

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQPE(const nb::module_& m) {
  nb::class_<bench::Phase>(m, "Phase",
                           "An exact phase in turns modulo one turn.")
      .def(nb::init<uint64_t, uint64_t>(), nb::kw_only(), "numerator"_a,
           "denominator"_a)
      .def_prop_ro("numerator", &bench::Phase::numerator,
                   "The reduced numerator.")
      .def_prop_ro("denominator", &bench::Phase::denominator,
                   "The reduced denominator.");

  nb::enum_<bench::QPEMethod>(
      m, "Method",
      "Full-register or iterative measurement-and-feed-forward method.")
      .value("STANDARD", bench::QPEMethod::Standard)
      .value("ITERATIVE", bench::QPEMethod::Iterative);

  nb::class_<bench::QPEOptions>(m, "Options", "Parameters for a QPE benchmark.")
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
                  "fractions.Fraction | Phase, method: Method = ...) -> "
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
      .def(
          "generate",
          [](const bench::QPE& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
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

} // namespace mqt::bindings
