/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/GHZ.hpp"
#include "bench/JSON.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>

namespace mqt::bindings {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerGHZ(const nb::module_& m) {
  nb::enum_<bench::GHZTopology>(m, "Topology",
                                "Entangling topology for GHZ preparation.")
      .value("LINEAR", bench::GHZTopology::Linear)
      .value("STAR", bench::GHZTopology::Star);
  nb::enum_<bench::GHZBasis>(m, "Basis",
                             "Measurement basis for GHZ verification.")
      .value("Z", bench::GHZBasis::Z)
      .value("X", bench::GHZBasis::X);

  nb::class_<bench::GHZOptions>(m, "Options", "Parameters for a GHZ benchmark.")
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
      .def(
          "generate",
          [](const bench::GHZ& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
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
}

} // namespace mqt::bindings
