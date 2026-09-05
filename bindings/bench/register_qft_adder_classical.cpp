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
#include "bench/QFTAdderClassical.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQFTAdderClassical(const nb::module_& m) {
  nb::class_<bench::QFTAdderClassicalOptions>(
      m, "Options", "Parameters for a classical-input QFT adder benchmark.")
      .def(nb::init<std::string>(), nb::kw_only(), "addend"_a)
      .def_ro("addend", &bench::QFTAdderClassicalOptions::addend,
              "The big-endian classical addend.");

  auto qftAdder = nb::class_<bench::QFTAdderClassical>(
      m, "QFTAdderClassical",
      "A validated classical-input QFT adder benchmark.");
  qftAdder.def(nb::init<bench::QFTAdderClassicalOptions>(), "options"_a)
      .def_prop_ro("options", &bench::QFTAdderClassical::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QFTAdderClassical::output,
                   nb::rv_policy::reference_internal,
                   "The logical result register.")
      .def_prop_ro("expected_result", &bench::QFTAdderClassical::expectedResult,
                   nb::rv_policy::reference_internal,
                   "The deterministic big-endian result.")
      .def("probability", &bench::QFTAdderClassical::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::QFTAdderClassical::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::QFTAdderClassical& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::QFTAdderClassical& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::QFTAdderClassical& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::QFTAdderClassical& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::qftAdderClassicalFromInstanceSpecificationJSON,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json",
                  &bench::qftAdderClassicalFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
