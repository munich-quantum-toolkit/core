/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Grover.hpp"
#include "bench/JSON.hpp"

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/optional.h"    // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <optional>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerGrover(const nb::module_& m) {
  nb::class_<bench::GroverOptions>(m, "Options",
                                   "Parameters for a Grover benchmark.")
      .def(nb::init<std::string, std::optional<size_t>>(), nb::kw_only(),
           "marked_bitstring"_a, "iterations"_a = nb::none())
      .def_ro("marked_bitstring", &bench::GroverOptions::markedBitstring,
              "The big-endian marked outcome.")
      .def_ro("iterations", &bench::GroverOptions::iterations,
              "The iteration count, or ``None`` for automatic selection.");

  auto grover = nb::class_<bench::Grover>(
      m, "Grover", "A validated single-solution Grover benchmark.");
  grover.def("__init__", benchmarkInit(&bench::Grover::create), "options"_a)
      .def_prop_ro("options", &bench::Grover::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::Grover::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro("qubits", &bench::Grover::qubits,
                   "The number of search qubits.")
      .def("probability", benchmarkMethod(&bench::Grover::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::Grover::evaluate), "counts"_a,
           "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(grover, &bench::groverFromInstanceSpecificationJSON,
                      &bench::groverFromManifestJSON);
}

} // namespace mqt
