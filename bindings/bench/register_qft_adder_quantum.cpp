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
#include "bench/QFTAdderQuantum.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQFTAdderQuantum(const nb::module_& m) {
  nb::class_<bench::QFTAdderQuantumOptions>(
      m, "Options", "Parameters for a quantum-input QFT adder benchmark.")
      .def(nb::init<size_t>(), nb::kw_only(), "qubits"_a)
      .def_ro("qubits", &bench::QFTAdderQuantumOptions::qubits,
              "The number of qubits in each input register.");

  auto qftAdder = nb::class_<bench::QFTAdderQuantum>(
      m, "QFTAdderQuantum", "A validated quantum-input QFT adder benchmark.");
  qftAdder.def(nb::init<bench::QFTAdderQuantumOptions>(), "options"_a)
      .def_prop_ro("options", &bench::QFTAdderQuantum::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro(
          "output", &bench::QFTAdderQuantum::output,
          nb::rv_policy::reference_internal,
          "The logical output register, with the addend followed by the sum.")
      .def("probability", &bench::QFTAdderQuantum::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::QFTAdderQuantum::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::QFTAdderQuantum& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::QFTAdderQuantum& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::QFTAdderQuantum& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::QFTAdderQuantum& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::qftAdderQuantumFromInstanceSpecificationJSON,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::qftAdderQuantumFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
