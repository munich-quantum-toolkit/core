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
#include "bench/QAOA.hpp"

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
void registerQAOA(const nb::module_& m) {
  nb::class_<bench::QAOAOptions>(m, "Options",
                                 "Parameters for a QAOA benchmark for MaxCut.")
      .def(nb::init<size_t, size_t>(), nb::kw_only(), "qubits"_a, "depth"_a = 1)
      .def_ro(
          "qubits", &bench::QAOAOptions::qubits,
          "The cycle's number of vertices, at least three and representable "
          "as a signed 64-bit integer.")
      .def_ro("depth", &bench::QAOAOptions::depth,
              "The number of cost-mixer layers.");

  nb::class_<bench::QAOAEvaluation>(m, "Evaluation",
                                    "The sampled cut quality on a cycle graph.")
      .def_ro("mean_cut", &bench::QAOAEvaluation::meanCut,
              "The shot-weighted mean number of cut edges.")
      .def_ro("approximation_ratio", &bench::QAOAEvaluation::approximationRatio,
              "The mean cut divided by the cycle's maximum cut size.");

  auto qaoa =
      nb::class_<bench::QAOA>(m, "QAOA",
                              R"pb(A validated QAOA benchmark for MaxCut.

The circuit starts in :math:`|+\rangle^{\otimes n}` on an unweighted cycle.
Each layer applies :math:`e^{-i\gamma_\ell C}` followed by
:math:`e^{-i\beta_\ell B}`, where
:math:`C=\sum_{j=0}^{n-1}(1-Z_jZ_{(j+1)\bmod n})/2` and
:math:`B=\sum_{j=0}^{n-1} X_j`.

For :math:`d` layers, the generated program takes two arrays of :math:`d`
angles: ``gammas`` and ``betas``. Bind angles in radians with
:meth:`mqt.core.mlir.QCProgram.bind_parameters` before sampling or export.
Copy the program first to reuse it for other angles.

The benchmark does not run an optimizer. Evaluation measures cut quality,
not agreement with an ideal output distribution.)pb");
  qaoa.def("__init__", benchmarkInit(&bench::QAOA::create), "options"_a)
      .def_prop_ro("options", &bench::QAOA::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::QAOA::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def("cut_size", benchmarkMethod(&bench::QAOA::cutSize), "outcome"_a,
           "Count cycle edges whose measured endpoint bits differ.")
      .def("evaluate", benchmarkMethod(&bench::QAOA::evaluate), "counts"_a,
           "Compute the mean cut and its ratio to the maximum cut size.");
  registerBenchmarkIO(qaoa, &bench::qaoaFromInstanceSpecificationJSON,
                      &bench::qaoaFromManifestJSON);
}

} // namespace mqt
