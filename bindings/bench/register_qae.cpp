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
#include "bench/QAE.hpp"

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"    // NOLINT(misc-include-cleaner)
#include "nanobind/stl/map.h"         // NOLINT(misc-include-cleaner)
#include "nanobind/stl/optional.h"    // NOLINT(misc-include-cleaner)
#include "nanobind/stl/pair.h"        // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)
#include "nanobind/stl/vector.h"      // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mqt {
namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQAE(nb::module_& m) {
  nb::class_<bench::QAEProblem>(
      m, "Problem",
      "State preparation and oracle for quantum amplitude estimation.")
      .def(nb::init<std::vector<std::string>, double>(), nb::kw_only(),
           "good_states"_a, "amplitude"_a)
      .def_ro(
          "good_states", &bench::QAEProblem::goodStates,
          "The nonempty set of distinct big-endian good states. All have the "
          "same width (1-63 bits), and at least one basis state must be bad.")
      .def_ro(
          "amplitude", &bench::QAEProblem::amplitude,
          "The total good-state probability in [0, 1] before amplification.");
  nb::class_<bench::QAEOptions>(m, "Options",
                                R"pb(Parameters for one QAE circuit.

Supply exactly one of precision and iterations.)pb")
      .def(nb::init<bench::QAEProblem, std::optional<size_t>,
                    std::optional<size_t>>(),
           nb::kw_only(), "problem"_a, "precision"_a = nb::none(),
           "iterations"_a = nb::none())
      .def_ro("problem", &bench::QAEOptions::problem,
              "The state preparation and good-state set.")
      .def_ro("precision", &bench::QAEOptions::precision,
              "The number of phase bits (1-31) for canonical QAE.")
      .def_ro(
          "iterations", &bench::QAEOptions::iterations,
          R"pb(The Grover power from 0 through :math:`2^{31}-1` for an amplification experiment.)pb");
  auto qae = nb::class_<bench::QAE>(
      m, "QAE", R"pb(A validated quantum amplitude estimation benchmark.

State preparation distributes each class's probability uniformly over its basis
states. A phase-estimation circuit returns the phase bits. An amplification
circuit returns the data bits for classification against the good-state set.)pb");
  qae.def("__init__", benchmarkInit(&bench::QAE::create), "options"_a)
      .def_prop_ro("options", &bench::QAE::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QAE::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("probability", benchmarkOutcome(&bench::QAE::probability),
           "outcome"_a, "Return the ideal probability of an outcome.")
      .def("evaluate", benchmarkMethod(&bench::QAE::evaluate), "counts"_a,
           "Compare sampled counts with the ideal distribution.");
  registerBenchmarkIO(qae, &bench::qaeFromInstanceSpecificationJSON,
                      &bench::qaeFromManifestJSON);

  using Runner = std::function<bench::Counts(const bench::QAE&)>;
  m.def(
      "phase_estimation",
      [](const bench::QAEProblem& problem, const Runner& run,
         size_t precision) {
        return runBenchmarkWorkflow(run, [&](bench::QAERun callback) {
          return bench::phaseEstimation(problem, precision, callback);
        });
      },
      "problem"_a, "run"_a, nb::kw_only(), "precision"_a,
      R"pb(Estimate the good-state probability using quantum phase estimation.

The callback takes a QAE instance with the requested precision (1-31 bits) and
returns its counts. The estimate uses the most frequent phase bin after
combining opposite phases. Ties select the lower probability.)pb");
  m.def(
      "maximum_likelihood",
      [](const bench::QAEProblem& problem, const Runner& run, size_t rounds) {
        return runBenchmarkWorkflow(run, [&](bench::QAERun callback) {
          return bench::maximumLikelihood(problem, rounds, callback);
        });
      },
      "problem"_a, "run"_a, nb::kw_only(), "rounds"_a,
      R"pb(Estimate the good-state probability by maximizing the joint binomial likelihood.

The callback takes a QAE instance and returns its counts. Choose 1-31 rounds.
The Grover powers are 1 through ``2**(rounds - 1)`` in powers of two. Each
experiment may use a different number of shots. A single round can give
multiple equally likely estimates.)pb");
  m.def(
      "iterative",
      [](const bench::QAEProblem& problem, const Runner& run, double epsilon,
         double confidence, size_t maxBatches) {
        return runBenchmarkWorkflow(run, [&](bench::QAERun callback) {
          return bench::iterativeAmplitudeEstimation(
              problem, epsilon, confidence, maxBatches, callback);
        });
      },
      "problem"_a, "run"_a, nb::kw_only(), "epsilon"_a, "confidence"_a = 0.95,
      "max_batches"_a = 100,
      R"pb(Estimate a probability interval by adapting the Grover power to measured counts.

The callback takes a QAE instance and returns an independent, fixed-size batch
of shots. The midpoint of the returned bounds estimates the probability within
epsilon at the requested confidence under ideal sampling. Choose epsilon in
(0, 0.5), confidence in (0, 1), and a positive batch budget.

Raises:
    ValueError: The inputs are invalid, the confidence intervals are disjoint,
        or the batch budget is exhausted.)pb");
}
} // namespace mqt
