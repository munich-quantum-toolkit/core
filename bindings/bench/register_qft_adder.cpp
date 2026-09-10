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
#include "bench/QFTAdder.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/optional.h"    // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQFTAdder(const nb::module_& m) {
  nb::enum_<bench::QFTAdderMethod>(m, "Method",
                                   "How the addend enters the circuit.")
      .value("REGISTER", bench::QFTAdderMethod::Register)
      .value("CONSTANT", bench::QFTAdderMethod::Constant);
  nb::enum_<bench::QFTAdderOverflow>(m, "Overflow",
                                     "Wrap the sum or retain carry.")
      .value("WRAP", bench::QFTAdderOverflow::Wrap)
      .value("CARRY", bench::QFTAdderOverflow::Carry);
  nb::class_<bench::QFTAdderOptions>(m, "Options",
                                     "Parameters for a QFT adder.")
      .def(nb::init<std::string, std::string, bench::QFTAdderMethod,
                    bench::QFTAdderOverflow>(),
           nb::kw_only(), "addend"_a, "accumulator"_a,
           "method"_a = bench::QFTAdderMethod::Register,
           "overflow"_a = bench::QFTAdderOverflow::Wrap)
      .def_ro(
          "addend", &bench::QFTAdderOptions::addend,
          R"pb(Big-endian addend; register inputs also accept ``+`` for a :math:`|+\rangle` qubit.)pb")
      .def_ro("accumulator", &bench::QFTAdderOptions::accumulator,
              "Binary accumulator with the same width as the addend.")
      .def_ro("method", &bench::QFTAdderOptions::method,
              "Register or constant addition.")
      .def_ro("overflow", &bench::QFTAdderOptions::overflow,
              "Wrap or carry behavior.");

  auto qftAdder =
      nb::class_<bench::QFTAdder>(m, "QFTAdder",
                                  R"pb(A validated QFT adder benchmark.

Register addition uses controlled phases and returns the addend followed by the
sum. Constant addition compiles the addend into phases and returns only the sum.
Wrap mode computes :math:`(a + b) \bmod 2^n`; carry mode retains one extra sum bit.
All strings are big-endian, and leading zeros determine the operand width.

Register addends may contain ``+`` for independent :math:`|+\rangle` qubits.
The accumulator and constant addends must be binary. The circuit follows
Draper's register addition and its constant-input Fourier specialization.)pb");
  qftAdder.def(nb::init<bench::QFTAdderOptions>(), "options"_a)
      .def_prop_ro("options", &bench::QFTAdder::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QFTAdder::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro(
          "expected_result", &bench::QFTAdder::expectedResult,
          "The unique logical outcome, or ``None`` for a superposed addend.")
      .def("probability", &bench::QFTAdder::probability, "outcome"_a,
           "Return the ideal probability of an outcome.")
      .def("evaluate", &bench::QFTAdder::evaluate, "counts"_a,
           "Compare sampled counts with the ideal distribution.")
      .def(
          "generate",
          [](const bench::QFTAdder& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::QFTAdder& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::QFTAdder& value) {
            return bench::toManifestJSON(value);
          },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::QFTAdder& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::qftAdderFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::qftAdderFromManifestJSON,
                  "json"_a, nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
