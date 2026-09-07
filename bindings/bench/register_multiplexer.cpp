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
#include "bench/Multiplexer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerMultiplexer(const nb::module_& m) {
  nb::class_<bench::MultiplexerOptions>(
      m, "Options", "Parameters for a quantum multiplexer benchmark.")
      .def(nb::init<size_t>(), nb::kw_only(), "qubits"_a)
      .def_ro("qubits", &bench::MultiplexerOptions::qubits,
              "The total number of control and target qubits.");

  auto multiplexer = nb::class_<bench::Multiplexer>(
      m, "Multiplexer", "A validated quantum multiplexer benchmark.");
  multiplexer.def(nb::init<bench::MultiplexerOptions>(), "options"_a)
      .def_prop_ro("options", &bench::Multiplexer::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::Multiplexer::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::Multiplexer::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::Multiplexer::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::Multiplexer& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::Multiplexer& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::Multiplexer& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::Multiplexer& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::multiplexerFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::multiplexerFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
