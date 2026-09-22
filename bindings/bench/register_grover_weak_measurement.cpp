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
#include "bench/WeakMeasurementGrover.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/optional.h"    // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include <optional>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerWeakMeasurementGrover(const nb::module_& m) {
  nb::class_<bench::WeakMeasurementGroverOptions>(
      m, "Options", "Parameters for a weak-measurement Grover benchmark.")
      .def(nb::init<std::string, std::optional<double>>(), nb::kw_only(),
           "marked_bitstring"_a, "measurement_strength"_a = nb::none())
      .def_ro("marked_bitstring",
              &bench::WeakMeasurementGroverOptions::markedBitstring,
              "The big-endian marked outcome.")
      .def_ro(
          "measurement_strength",
          &bench::WeakMeasurementGroverOptions::measurementStrength,
          R"pb(The :math:`\kappa`-measurement strength, or ``None`` for :math:`2^{-n/2}`.)pb");

  auto grover = nb::class_<bench::WeakMeasurementGrover>(
      m, "Grover",
      R"pb(A validated weak-measurement Grover benchmark.

The benchmark prepares a uniform state and applies one Grover iteration before
each :math:`\kappa`-measurement. The measurement computes the predicate with
:math:`O_\chi`, applies the controlled :math:`R_\kappa` rotation, uncomputes
:math:`O_\chi`, and measures the probe. An outcome of :math:`0` continues the
loop; :math:`1` exits with the marked state.

By default, the measurement strength is :math:`\kappa=2^{-n/2}` for :math:`n`
search qubits. The accepted range :math:`0<\kappa\leq 2^{-n/2}` follows the
paper's robustness bound.)pb");
  grover.def(nb::init<bench::WeakMeasurementGroverOptions>(), "options"_a)
      .def_prop_ro("options", &bench::WeakMeasurementGrover::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::WeakMeasurementGrover::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro("qubits", &bench::WeakMeasurementGrover::qubits,
                   "The number of search qubits.")
      .def("probability", &bench::WeakMeasurementGrover::probability,
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::WeakMeasurementGrover::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::WeakMeasurementGrover& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::WeakMeasurementGrover& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::WeakMeasurementGrover& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::WeakMeasurementGrover& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::weakMeasurementGroverFromInstanceSpecificationJSON,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json",
                  &bench::weakMeasurementGroverFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
