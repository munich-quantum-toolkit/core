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
#include "bench/Teleportation.hpp"

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerTeleportation(const nb::module_& m) {
  auto teleportation = nb::class_<bench::Teleportation>(
      m, "Teleportation",
      R"pb(A validated quantum teleportation benchmark.

The circuit teleports the fixed :math:`|+\rangle` state and measures Bob's qubit
in the :math:`X` basis. The one-bit result is zero on success; Alice's
measurements stay internal.)pb");
  teleportation.def(nb::init<>())
      .def_prop_ro("output", &bench::Teleportation::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkMethod(&bench::Teleportation::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::Teleportation::evaluate),
           "counts"_a, "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(teleportation,
                      &bench::teleportationFromInstanceSpecificationJSON,
                      &bench::teleportationFromManifestJSON);
}

} // namespace mqt
