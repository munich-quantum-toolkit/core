/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"
#include "bench/JSON.hpp"

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerBV(const nb::module_& m) {
  nb::enum_<bench::BVMethod>(
      m, "Method", "Static allocation or dynamic measurement and qubit reuse.")
      .value("STATIC", bench::BVMethod::Static)
      .value("DYNAMIC", bench::BVMethod::Dynamic);

  nb::class_<bench::BVOptions>(
      m, "Options", "Parameters for a Bernstein--Vazirani benchmark.")
      .def(nb::init<std::string, bench::BVMethod>(), nb::kw_only(),
           "hidden_bitstring"_a, "method"_a = bench::BVMethod::Static)
      .def_ro("hidden_bitstring", &bench::BVOptions::hiddenBitstring,
              "The big-endian hidden bitstring.")
      .def_ro("method", &bench::BVOptions::method, "The circuit method.");

  auto bv = nb::class_<bench::BV>(m, "BV",
                                  "A validated Bernstein--Vazirani benchmark.");
  bv.def("__init__", benchmarkInit(&bench::BV::create), "options"_a)
      .def_prop_ro("options", &bench::BV::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::BV::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkMethod(&bench::BV::probability), "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::BV::evaluate), "counts"_a,
           "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(bv, &bench::bvFromInstanceSpecificationJSON,
                      &bench::bvFromManifestJSON);
}

} // namespace mqt
