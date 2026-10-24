/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/GHZ.hpp"
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
void registerGHZ(const nb::module_& m) {
  nb::enum_<bench::GHZTopology>(m, "Topology",
                                "Entangling topology for GHZ preparation.")
      .value("LINEAR", bench::GHZTopology::Linear)
      .value("STAR", bench::GHZTopology::Star);
  nb::enum_<bench::GHZBasis>(m, "Basis",
                             "Measurement basis for GHZ verification.")
      .value("Z", bench::GHZBasis::Z)
      .value("X", bench::GHZBasis::X);

  nb::class_<bench::GHZOptions>(m, "Options", "Parameters for a GHZ benchmark.")
      .def(nb::init<size_t, bench::GHZTopology, bench::GHZBasis>(),
           nb::kw_only(), "qubits"_a, "topology"_a = bench::GHZTopology::Linear,
           "basis"_a = bench::GHZBasis::Z)
      .def_ro("qubits", &bench::GHZOptions::qubits, "The number of qubits.")
      .def_ro("topology", &bench::GHZOptions::topology,
              "The entangling topology.")
      .def_ro("basis", &bench::GHZOptions::basis, "The measurement basis.");

  auto ghz = nb::class_<bench::GHZ>(m, "GHZ", "A validated GHZ benchmark.");
  ghz.def("__init__", benchmarkInit(&bench::GHZ::create), "options"_a)
      .def_prop_ro("options", &bench::GHZ::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::GHZ::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkMethod(&bench::GHZ::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::GHZ::evaluate), "counts"_a,
           "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(ghz, &bench::ghzFromInstanceSpecificationJSON,
                      &bench::ghzFromManifestJSON);
}

} // namespace mqt
