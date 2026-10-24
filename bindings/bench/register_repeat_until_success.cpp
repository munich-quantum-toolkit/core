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

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

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
      .def("__init__", benchmarkInit(&bench::RepeatUntilSuccess::create),
           "options"_a = bench::RepeatUntilSuccessOptions{})
      .def_prop_ro("options", &bench::RepeatUntilSuccess::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::RepeatUntilSuccess::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability",
           benchmarkMethod(&bench::RepeatUntilSuccess::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::RepeatUntilSuccess::evaluate),
           "counts"_a, "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(repeatUntilSuccess,
                      &bench::repeatUntilSuccessFromInstanceSpecificationJSON,
                      &bench::repeatUntilSuccessFromManifestJSON);
}

} // namespace mqt
