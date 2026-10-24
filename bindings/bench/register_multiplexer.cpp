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
#include "bench/Multiplexer.hpp"

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
void registerMultiplexer(const nb::module_& m) {
  nb::class_<bench::MultiplexerOptions>(
      m, "Options", "Parameters for a quantum multiplexer benchmark.")
      .def(nb::init<size_t>(), nb::kw_only(), "qubits"_a)
      .def_ro("qubits", &bench::MultiplexerOptions::qubits,
              "The total number of control and target qubits.");

  auto multiplexer = nb::class_<bench::Multiplexer>(
      m, "Multiplexer", "A validated quantum multiplexer benchmark.");
  multiplexer
      .def("__init__", benchmarkInit(&bench::Multiplexer::create), "options"_a)
      .def_prop_ro("options", &bench::Multiplexer::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::Multiplexer::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkMethod(&bench::Multiplexer::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::Multiplexer::evaluate),
           "counts"_a, "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(multiplexer,
                      &bench::multiplexerFromInstanceSpecificationJSON,
                      &bench::multiplexerFromManifestJSON);
}

} // namespace mqt
