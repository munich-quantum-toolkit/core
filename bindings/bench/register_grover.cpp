/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Grover.hpp"
#include "bench/JSON.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/optional.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <optional>
#include <string>

namespace mqt::bindings {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerGrover(const nb::module_& m) {
  nb::class_<bench::GroverOptions>(m, "Options",
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
      .def(
          "generate",
          [](const bench::Grover& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
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
}

} // namespace mqt::bindings
