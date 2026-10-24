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
#include "bench/GroverWeakMeasurement.hpp"

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/optional.h"    // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include <optional>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerGroverWeakMeasurement(const nb::module_& m) {
  nb::class_<bench::GroverWeakMeasurementOptions>(
      m, "Options", "Parameters for a weak-measurement Grover benchmark.")
      .def(nb::init<std::string, std::optional<double>>(), nb::kw_only(),
           "marked_bitstring"_a, "measurement_strength"_a = nb::none())
      .def_ro("marked_bitstring",
              &bench::GroverWeakMeasurementOptions::markedBitstring,
              "The big-endian marked outcome with 2 through 2044 bits.")
      .def_ro(
          "measurement_strength",
          &bench::GroverWeakMeasurementOptions::measurementStrength,
          R"pb(The finite strength :math:`0<\kappa\leq 2^{-n/2}`, or ``None`` for the upper bound.)pb");

  auto grover = nb::class_<bench::GroverWeakMeasurement>(
      m, "Grover",
      R"pb(A validated weak-measurement Grover benchmark.

The benchmark prepares a uniform state and applies one Grover iteration before
each :math:`\kappa`-measurement. The measurement computes the predicate with
:math:`O_\chi`, applies the controlled :math:`R_\kappa` rotation, uncomputes
:math:`O_\chi`, and measures the probe. An outcome of :math:`0` continues the
loop; :math:`1` exits with the marked state.

By default, the measurement strength is :math:`\kappa=2^{-n/2}` for :math:`n`
search qubits. The accepted range :math:`0<\kappa\leq 2^{-n/2}` follows the
paper's robustness bound.)pb");
  grover
      .def("__init__", benchmarkInit(&bench::GroverWeakMeasurement::create),
           "options"_a)
      .def_prop_ro("options", &bench::GroverWeakMeasurement::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::GroverWeakMeasurement::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro("qubits", &bench::GroverWeakMeasurement::qubits,
                   "The number of search qubits.")
      .def("probability",
           benchmarkMethod(&bench::GroverWeakMeasurement::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::GroverWeakMeasurement::evaluate),
           "counts"_a, "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(
      grover, &bench::groverWeakMeasurementFromInstanceSpecificationJSON,
      &bench::groverWeakMeasurementFromManifestJSON);
}

} // namespace mqt
