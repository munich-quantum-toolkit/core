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
#include "bench/RepeatUntilSuccess.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <cstddef>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerRepeatUntilSuccess(const nb::module_& m) {
  nb::class_<bench::RepeatUntilSuccessOptions>(
      m, "Options",
      "Parameters for a Pauli-string repeat-until-success benchmark.")
      .def(nb::init<size_t>(), nb::kw_only(), "data_qubits"_a = 1)
      .def_ro("data_qubits", &bench::RepeatUntilSuccessOptions::dataQubits,
              "The number of data qubits, excluding the ancilla.");
  auto repeatUntilSuccess = nb::class_<bench::RepeatUntilSuccess>(
      m, "RepeatUntilSuccess",
      R"pb(A validated Pauli-string repeat-until-success benchmark.

The circuit applies :math:`(I + i\sqrt{2}X^{\otimes n}) / \sqrt{3}`. Each
attempt measures an ancilla prepared from :math:`|0\rangle` and retries on
failure. After success, return the parity of a
:math:`Y \otimes X^{\otimes(n-1)}` measurement on the data qubits.)pb");
  repeatUntilSuccess
      .def(nb::init<bench::RepeatUntilSuccessOptions>(),
           "options"_a = bench::RepeatUntilSuccessOptions{})
      .def_prop_ro("options", &bench::RepeatUntilSuccess::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::RepeatUntilSuccess::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::RepeatUntilSuccess::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::RepeatUntilSuccess::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::RepeatUntilSuccess& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::RepeatUntilSuccess& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::RepeatUntilSuccess& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::RepeatUntilSuccess& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::repeatUntilSuccessFromInstanceSpecificationJSON,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json",
                  &bench::repeatUntilSuccessFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
