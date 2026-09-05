/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/ControlledMultiplicationModuloN.hpp"
#include "bench/JSON.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerControlledMultiplicationModuloN(const nb::module_& m) {
  nb::class_<bench::ControlledMultiplicationModuloNOptions>(
      m, "Options",
      "Parameters for a controlled multiplication modulo N benchmark.")
      .def(nb::init<std::string, std::string>(), nb::kw_only(), "multiplier"_a,
           "modulus"_a)
      .def_ro("multiplier",
              &bench::ControlledMultiplicationModuloNOptions::multiplier,
              "The big-endian classical multiplier.")
      .def_ro("modulus",
              &bench::ControlledMultiplicationModuloNOptions::modulus,
              "The canonical big-endian modulus.");

  auto controlledMultiplicationModuloN =
      nb::class_<bench::ControlledMultiplicationModuloN>(
          m, "ControlledMultiplicationModuloN",
          "A validated controlled multiplication modulo N benchmark.");
  controlledMultiplicationModuloN
      .def(nb::init<bench::ControlledMultiplicationModuloNOptions>(),
           "options"_a)
      .def_prop_ro("options", &bench::ControlledMultiplicationModuloN::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::ControlledMultiplicationModuloN::output,
                   nb::rv_policy::reference_internal,
                   "The logical control, multiplicand, and accumulator output.")
      .def("probability", &bench::ControlledMultiplicationModuloN::probability,
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::ControlledMultiplicationModuloN::evaluate,
           "counts"_a, "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::ControlledMultiplicationModuloN& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::ControlledMultiplicationModuloN& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::ControlledMultiplicationModuloN& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::ControlledMultiplicationModuloN& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static(
          "from_instance_specification_json",
          &bench::controlledMultiplicationModuloNFromInstanceSpecificationJSON,
          "json"_a, nb::kw_only(), "source"_a = "<instance-specification>",
          "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json",
                  &bench::controlledMultiplicationModuloNFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
