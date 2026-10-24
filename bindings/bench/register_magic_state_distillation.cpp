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
void registerMagicStateDistillation(const nb::module_& m) {
  nb::class_<bench::MagicStateDistillationOptions>(
      m, "Options",
      "Parameters for concatenated 15-to-1 magic-state distillation.")
      .def(nb::init<size_t>(), nb::kw_only(), "levels"_a = 1)
      .def_ro("levels", &bench::MagicStateDistillationOptions::levels,
              R"pb(Positive number of concatenated levels.

The qubit count (five per level) must fit signed 64-bit circuit dimensions.)pb");
  auto magicStateDistillation = nb::class_<bench::MagicStateDistillation>(
      m, "MagicStateDistillation",
      R"pb(A concatenated 15-to-1 magic-state distillation benchmark.

Each block uses Litinski's five-qubit 15-to-1 circuit. The first level uses
ideal :math:`\pi/8` Pauli rotations; later levels implement these rotations
with the preceding level's retained :math:`T^\dagger|+\rangle` states.
The circuit uses ``5 * levels`` qubits and executes ``15**levels`` leaf
rotations per shot. Backend and resource limits apply.

Bit 1 flags any rejected block; bit 0 checks the retained root state against
:math:`T^\dagger|+\rangle`. Ideal output is ``00``. All blocks execute once,
regardless of rejection. The benchmark models ideal logical circuits without
input noise, physical error correction, or retries.)pb");
  magicStateDistillation
      .def("__init__", benchmarkInit(&bench::MagicStateDistillation::create),
           "options"_a = bench::MagicStateDistillationOptions{})
      .def_prop_ro("options", &bench::MagicStateDistillation::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::MagicStateDistillation::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability",
           benchmarkMethod(&bench::MagicStateDistillation::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate",
           benchmarkMethod(&bench::MagicStateDistillation::evaluate),
           "counts"_a, "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(
      magicStateDistillation,
      &bench::magicStateDistillationFromInstanceSpecificationJSON,
      &bench::magicStateDistillationFromManifestJSON);
}

} // namespace mqt
