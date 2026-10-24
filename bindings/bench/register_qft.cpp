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
#include "bench/QFT.hpp"

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
void registerQFT(const nb::module_& m) {
  nb::enum_<bench::QFTMethod>(
      m, "Method",
      "Full-register or semiclassical measurement-and-feed-forward method.")
      .value("STANDARD", bench::QFTMethod::Standard)
      .value("SEMICLASSICAL", bench::QFTMethod::Semiclassical);

  nb::class_<bench::QFTOptions>(m, "Options", "Parameters for a QFT benchmark.")
      .def(nb::init<size_t, size_t, bench::QFTMethod>(), nb::kw_only(),
           "qubits"_a, "period_exponent"_a,
           "method"_a = bench::QFTMethod::Standard)
      .def_ro("qubits", &bench::QFTOptions::qubits,
              "The number of transformed qubits.")
      .def_ro("period_exponent", &bench::QFTOptions::periodExponent,
              R"pb(The exponent :math:`k` of the input period :math:`2^k`.)pb")
      .def_ro("method", &bench::QFTOptions::method, "The circuit method.");

  auto qft = nb::class_<bench::QFT>(m, "QFT", "A validated QFT benchmark.");
  qft.def("__init__", benchmarkInit(&bench::QFT::create), "options"_a)
      .def_prop_ro("options", &bench::QFT::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QFT::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkMethod(&bench::QFT::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::QFT::evaluate), "counts"_a,
           "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(qft, &bench::qftFromInstanceSpecificationJSON,
                      &bench::qftFromManifestJSON);
}

} // namespace mqt
