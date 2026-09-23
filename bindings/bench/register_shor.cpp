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
#include "bench/JSON.hpp"
#include "bench/Shor.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"    /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/map.h"         /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/optional.h"    /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/pair.h"        /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      /// NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" /// NOLINT(misc-include-cleaner)

#include <cstddef>
#include <cstdint>
#include <functional>

namespace mqt {
namespace nb = nanobind;
using namespace nb::literals;

/// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerShor(nb::module_& m) {
  nb::class_<bench::ShorOptions>(
      m, "Options", "Parameters for semiclassical Shor order finding.")
      .def(nb::init<uint64_t, uint64_t>(), nb::kw_only(), "number"_a,
           "base"_a = 2)
      .def_ro("number", &bench::ShorOptions::number,
              "The odd modulus, at most 2**31 - 1.")
      .def_ro("base", &bench::ShorOptions::base,
              "The base, coprime to the modulus.");
  nb::class_<bench::ShorEvaluation>(
      m, "Evaluation", "Verified factors recovered from measured phases.")
      .def_ro("success_probability", &bench::ShorEvaluation::successProbability,
              "The fraction of shots that independently yield a verified "
              "factor pair.")
      .def_ro("factors", &bench::ShorEvaluation::factors,
              "A sorted factor pair, or ``None``.");
  nb::class_<bench::Shor>(
      m, "Shor", "Semiclassical order finding with one reused query qubit.")
      .def(nb::init<bench::ShorOptions>(), "options"_a)
      .def_prop_ro("options", &bench::Shor::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro(
          "output", &bench::Shor::output, nb::rv_policy::reference_internal,
          "The big-endian phase register with twice the modulus bit width.")
      .def("evaluate", &bench::Shor::evaluate, "counts"_a,
           "Recover factors using exact continued fractions and verify them by "
           "division.")
      .def(
          "generate",
          [](const bench::Shor& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const bench::Shor& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const bench::Shor& value) { return bench::toManifestJSON(value); },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const bench::Shor& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json",
                  &bench::shorFromInstanceSpecificationJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", &bench::shorFromManifestJSON, "json"_a,
                  nb::kw_only(), "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
  nb::enum_<bench::FactorStatus>(m, "FactorStatus",
                                 "Outcome of a factoring workflow.")
      .value("SUCCESS", bench::FactorStatus::Success)
      .value("PRIME", bench::FactorStatus::Prime)
      .value("ATTEMPTS_EXHAUSTED", bench::FactorStatus::AttemptsExhausted);
  nb::class_<bench::FactorResult>(m, "FactorResult",
                                  "Result of a bounded factoring workflow.")
      .def_ro("status", &bench::FactorResult::status)
      .def_ro("factors", &bench::FactorResult::factors,
              "A verified sorted pair, or ``None``.")
      .def_ro("attempts", &bench::FactorResult::attempts,
              "The number of attempted bases.");
  m.def(
      "factor",
      [](uint64_t number,
         const std::function<bench::Counts(const bench::Shor&)>& run,
         size_t maxAttempts, uint64_t seed) {
        return bench::factor(number, run,
                             {.maxAttempts = maxAttempts, .seed = seed});
      },
      "number"_a, "run"_a, nb::kw_only(), "max_attempts"_a = 16, "seed"_a = 0,
      R"pb(Find one nontrivial factor pair with a bounded number of attempted bases.

The callback accepts a :class:`Shor` instance and returns counts. It owns device
selection, shots, and execution seeds. Even numbers, primes, and perfect powers
are handled classically. Callback failures and invalid counts propagate.)pb");
}

} // namespace mqt
