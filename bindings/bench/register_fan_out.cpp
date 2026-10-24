/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/FanOut.hpp"
#include "bench/JSON.hpp"

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
void registerFanOut(const nb::module_& m) {
  nb::class_<bench::FanOutOptions>(
      m, "Options", "Parameters for a quantum fan-out benchmark.")
      .def(nb::init<size_t>(), nb::kw_only(), "controls"_a)
      .def_ro("controls", &bench::FanOutOptions::controls,
              R"pb(The number of control qubits, :math:`n\in[1, 1073]`.)pb");

  auto fanOut =
      nb::class_<bench::FanOut>(m, "FanOut",
                                R"pb(A validated quantum fan-out benchmark.

The benchmark prepares uniform controls and a target in :math:`|+\rangle`.
Fan-out distributes controlled phase gates :math:`P(\pi/2)` across the
target and clean ancillas, then uncomputes the ancillas. A final Hadamard on
the target makes the phase observable.

For :math:`n` controls, the circuit uses :math:`2n` qubits. The logical
output contains the controls followed by the target as its least-significant
bit. For a control string of Hamming weight :math:`w`, the target is zero
with conditional probability :math:`\cos^2(\pi w/4)`.)pb");
  fanOut.def("__init__", benchmarkInit(&bench::FanOut::create), "options"_a)
      .def_prop_ro("options", &bench::FanOut::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::FanOut::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkMethod(&bench::FanOut::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::FanOut::evaluate), "counts"_a,
           "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(fanOut, &bench::fanOutFromInstanceSpecificationJSON,
                      &bench::fanOutFromManifestJSON);
}

} // namespace mqt
