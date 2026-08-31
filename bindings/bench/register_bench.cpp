/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>   // NOLINT(misc-include-cleaner)

namespace mqt {

namespace nb = nanobind;

namespace bindings {
void registerBV(const nb::module_& m);
void registerGHZ(const nb::module_& m);
void registerGrover(const nb::module_& m);
void registerQFT(const nb::module_& m);
void registerQPE(const nb::module_& m);
} // namespace bindings

// The nanobind module macro requires its module handle by value.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = "Typed benchmark instances and analytic references.";

  nb::class_<bench::Output>(m, "Output",
                            "One logical classical output register.")
      .def_ro("name", &bench::Output::name, "The register name.")
      .def_ro("width", &bench::Output::width,
              "The number of big-endian outcome bits.");

  nb::class_<bench::Evaluation>(
      m, "Evaluation", "The result of comparing counts with a reference.")
      .def_ro("total_variation_distance",
              &bench::Evaluation::totalVariationDistance,
              "The total variation distance from the ideal distribution.")
      .def_ro("squared_hellinger_fidelity",
              &bench::Evaluation::squaredHellingerFidelity,
              "The squared Hellinger fidelity with the ideal distribution.")
      .def_ro("success_probability", &bench::Evaluation::successProbability,
              "The observed success probability, when defined.");

  const nb::module_ bv = m.def_submodule(
      "bv", "Bernstein--Vazirani benchmark instances and options.");
  bindings::registerBV(bv);

  const nb::module_ ghz =
      m.def_submodule("ghz", "GHZ benchmark instances and options.");
  bindings::registerGHZ(ghz);

  const nb::module_ grover =
      m.def_submodule("grover", "Grover benchmark instances and options.");
  bindings::registerGrover(grover);

  const nb::module_ qft =
      m.def_submodule("qft", "QFT benchmark instances and options.");
  bindings::registerQFT(qft);

  const nb::module_ qpe =
      m.def_submodule("qpe", "QPE benchmark instances and options.");
  bindings::registerQPE(qpe);
}

} // namespace mqt
