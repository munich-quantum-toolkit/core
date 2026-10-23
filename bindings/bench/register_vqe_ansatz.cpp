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
#include "bench/VQEAnsatz.hpp"

#include "register_benchmark.hpp"

#include "nanobind/nanobind.h"

#include <cstddef>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVQEAnsatz(const nb::module_& m) {
  nb::class_<bench::VQEAnsatzOptions>(m, "Options",
                                      "Parameters for a VQE ansatz benchmark.")
      .def(nb::init<size_t, size_t>(), nb::kw_only(), "qubits"_a, "depth"_a = 1)
      .def_ro("qubits", &bench::VQEAnsatzOptions::qubits,
              "The number of qubits, at least one.")
      .def_ro("depth", &bench::VQEAnsatzOptions::depth,
              "The number of rotation-entanglement layers.");

  auto vqeAnsatz = nb::class_<bench::VQEAnsatz>(
      m, "VQEAnsatz", R"pb(A validated VQE ansatz benchmark.

The RealAmplitudes circuit starts in :math:`|0\rangle^{\otimes n}`.
Each layer applies :math:`R_y` rotations to all qubits, followed by
:math:`\mathrm{CX}(0,1), \mathrm{CX}(1,2), \ldots, \mathrm{CX}(n-2,n-1)`
in that order. A final layer of :math:`R_y` rotations follows the last
entanglement layer.

For :math:`d` layers, the generated program takes one array, ``thetas``,
of :math:`n(d+1)` angles. Entry :math:`\ell n+q` controls the rotation
on qubit :math:`q` in rotation layer :math:`\ell`. Bind angles in radians
with :meth:`mqt.core.mlir.QCProgram.bind_parameters` before sampling or
export. Copy the program first to reuse it for other angles.

The benchmark supplies only the ansatz. It does not define a Hamiltonian,
run an optimizer, or evaluate energies.)pb");
  vqeAnsatz.def(nb::init<bench::VQEAnsatzOptions>(), "options"_a)
      .def_prop_ro("options", &bench::VQEAnsatz::options,
                   nb::rv_policy::reference_internal,
                   "The resolved benchmark parameters.")
      .def_prop_ro("output", &bench::VQEAnsatz::output,
                   nb::rv_policy::reference_internal,
                   "The logical output register.")
      .def_prop_ro("num_parameters", &bench::VQEAnsatz::numParameters,
                   "The number of rotation angles, including the final layer.");
  registerBenchmarkIO(vqeAnsatz, &bench::vqeAnsatzFromInstanceSpecificationJSON,
                      &bench::vqeAnsatzFromManifestJSON);
}

} // namespace mqt
