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

// forward declarations
void registerBV(const nb::module_& m);
void registerControlledMultiplicationModuloN(const nb::module_& m);
void registerGHZ(const nb::module_& m);
void registerGrover(const nb::module_& m);
void registerMultiplexer(const nb::module_& m);
void registerQFT(const nb::module_& m);
void registerQFTAdderClassical(const nb::module_& m);
void registerQFTAdderQuantum(const nb::module_& m);
void registerQPE(const nb::module_& m);
void registerRepeatUntilSuccess(const nb::module_& m);
void registerTeleportation(const nb::module_& m);

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
  registerBV(bv);

  const nb::module_ controlledMultiplicationModuloN = m.def_submodule(
      "controlled_multiplication_modulo_n",
      "Controlled multiplication modulo N benchmark instances and options.");
  registerControlledMultiplicationModuloN(controlledMultiplicationModuloN);

  const nb::module_ ghz =
      m.def_submodule("ghz", "GHZ benchmark instances and options.");
  registerGHZ(ghz);

  const nb::module_ grover =
      m.def_submodule("grover", "Grover benchmark instances and options.");
  registerGrover(grover);

  const nb::module_ multiplexer = m.def_submodule(
      "multiplexer", "Quantum multiplexer benchmark instances and options.");
  registerMultiplexer(multiplexer);

  const nb::module_ qft =
      m.def_submodule("qft", "QFT benchmark instances and options.");
  registerQFT(qft);

  const nb::module_ qftAdderClassical =
      m.def_submodule("qft_adder_classical",
                      "Classical-input QFT adder instances and options.");
  registerQFTAdderClassical(qftAdderClassical);

  const nb::module_ qftAdderQuantum = m.def_submodule(
      "qft_adder_quantum", "Quantum-input QFT adder instances and options.");
  registerQFTAdderQuantum(qftAdderQuantum);

  const nb::module_ qpe =
      m.def_submodule("qpe", "QPE benchmark instances and options.");
  registerQPE(qpe);

  const nb::module_ repeatUntilSuccess = m.def_submodule(
      "repeat_until_success", "Fixed repeat-until-success benchmark.");
  registerRepeatUntilSuccess(repeatUntilSuccess);

  const nb::module_ teleportation =
      m.def_submodule("teleportation", "Quantum teleportation benchmarks.");
  registerTeleportation(teleportation);
}

} // namespace mqt
