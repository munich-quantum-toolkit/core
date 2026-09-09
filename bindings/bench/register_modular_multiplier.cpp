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
#include "bench/ModularMultiplier.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>         // NOLINT(misc-include-cleaner)
#include <nanobind/stl/optional.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)

#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerModularMultiplier(const nb::module_& m) {
  nb::class_<bench::ModularMultiplierOptions>(
      m, "Options", R"pb(Parameters for a modular multiplier benchmark.)pb")
      .def(nb::init<std::string, std::string, std::string, char>(),
           nb::kw_only(), "multiplier"_a, "modulus"_a, "multiplicand"_a,
           "control"_a = '1')
      .def_ro("multiplier", &bench::ModularMultiplierOptions::multiplier,
              "The big-endian classical multiplier.")
      .def_ro("modulus", &bench::ModularMultiplierOptions::modulus,
              "The canonical big-endian modulus.")
      .def_ro(
          "multiplicand", &bench::ModularMultiplierOptions::multiplicand,
          R"pb(The big-endian multiplicand; ``+`` prepares :math:`|+\rangle`.)pb")
      .def_ro(
          "control", &bench::ModularMultiplierOptions::control,
          R"pb(The control input (``0``, ``1``, or ``+``); ``+`` prepares :math:`|+\rangle`.)pb");

  auto modularMultiplier = nb::class_<bench::ModularMultiplier>(
      m, "ModularMultiplier",
      R"pb(Compute :math:`c \cdot a \cdot x \bmod N` in an initially zero product register.

Here, :math:`c` is the control, :math:`a` is the classical multiplier,
:math:`x` is the multiplicand, and :math:`N` is the modulus.)pb");
  modularMultiplier
      .def(nb::init<bench::ModularMultiplierOptions>(), "options"_a)
      .def_prop_ro("options", &bench::ModularMultiplier::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::ModularMultiplier::output,
                   nb::rv_policy::reference_internal,
                   "The logical control, multiplicand, and accumulator output.")
      .def_prop_ro("expected_result", &bench::ModularMultiplier::expectedResult,
                   "The unique outcome, or ``None`` for superposed inputs.")
      .def("probability", &bench::ModularMultiplier::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::ModularMultiplier::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::ModularMultiplier& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::ModularMultiplier& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::ModularMultiplier& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::ModularMultiplier& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::modularMultiplierFromInstanceSpecificationJSON,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json",
                  &bench::modularMultiplierFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
