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
#include "bench/MagicStateDistillation.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" /// NOLINT(misc-include-cleaner)

#include <cstddef>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

/// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerMagicStateDistillation(const nb::module_& m) {
  nb::class_<bench::MagicStateDistillationOptions>(
      m, "Options",
      "Parameters for concatenated 15-to-1 magic-state distillation.")
      .def(nb::init<size_t>(), nb::kw_only(), "levels"_a = 1)
      .def_ro("levels", &bench::MagicStateDistillationOptions::levels,
              "Concatenated levels in [1, 4], using 15**levels qubits.");
  auto magicStateDistillation = nb::class_<bench::MagicStateDistillation>(
      m, "MagicStateDistillation",
      R"pb(Concatenated 15-to-1 Reed--Muller distillation.

Inputs are ideal :math:`|T\rangle = T|+\rangle` states.
Bit 1 flags any rejected block; bit 0 checks the retained root state in the T
basis. Ideal output is ``00``. Each level consumes the preceding level's
retained quantum outputs, using exactly ``15**levels`` qubits.)pb");
  magicStateDistillation
      .def(nb::init<bench::MagicStateDistillationOptions>(),
           "options"_a = bench::MagicStateDistillationOptions{})
      .def_prop_ro("options", &bench::MagicStateDistillation::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::MagicStateDistillation::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", &bench::MagicStateDistillation::probability,
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::MagicStateDistillation::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::MagicStateDistillation& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::MagicStateDistillation& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::MagicStateDistillation& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::MagicStateDistillation& value) {
            return bench::caseId(value);
          },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::magicStateDistillationFromInstanceSpecificationJSON,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json",
                  &bench::magicStateDistillationFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} /* namespace mqt */
