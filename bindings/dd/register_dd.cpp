/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/QuantumComputation.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// forward declarations
void registerVectorDDs(const nb::module_& m);
void registerMatrixDDs(const nb::module_& m);
void registerDDPackage(const nb::module_& m);

using Vector = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<1>>;
Vector getVector(const dd::vEdge& v, dd::fp threshold = 0.);

using Matrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>>;
Matrix getMatrix(const dd::mEdge& m, size_t numQubits, dd::fp threshold = 0.);

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  nb::module_::import_("mqt.core.ir");

  // Vector Decision Diagrams
  registerVectorDDs(m);

  // Matrix Decision Diagrams
  registerMatrixDDs(m);

  // DD Package
  registerDDPackage(m);

  m.def(
      "sample",
      [](const qc::QuantumComputation& qc, const size_t shots = 1024U,
         const size_t seed = 0U) { return dd::sample(qc, shots, seed); },
      "qc"_a, "shots"_a = 1024U, "seed"_a = 0U,
      R"pb(Sample from the output distribution of a quantum computation.

This function classically simulates the quantum computation and repeatedly samples from the output distribution.
It supports mid-circuit measurements, resets, and classical control.

Args:
    qc: The quantum computation.
    shots: The number of samples to take.
        If the quantum computation contains no mid-circuit measurements or resets, the circuit is simulated once and the samples are drawn from the final state.
        Otherwise, the circuit is simulated once for each sample.
        Defaults to 1024.
    seed: The seed for the random number generator.
        If set to a specific non-zero value, the simulation is deterministic.
        If set to 0, the RNG is randomly seeded.
        Defaults to 0.

Returns:
    A histogram of the samples.
    Each sample is a bitstring representing the measurement outcomes of the qubits in the quantum computation.
    The leftmost bit corresponds to the most significant qubit, that is, the qubit with the highest index (big-endian).
    If the circuit contains measurements, only the qubits that are actively measured are included in the output distribution.
    Otherwise, all qubits in the circuit are measured.)pb");

  m.def(
      "simulate_statevector",
      [](const qc::QuantumComputation& qc) {
        const auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        const auto in = makeZeroState(qc.getNqubits(), *dd);
        const auto sim = simulate(qc, in, *dd);
        return getVector(sim);
      },
      "qc"_a,
      R"pb(Simulate the quantum computation and return the final state vector.

This function classically simulates the quantum computation and returns the state vector of the final state.
It does not support measurements, resets, or classical control.

Since the state vector is guaranteed to be exponentially large in the number of qubits, this function is only suitable for small quantum computations.
Consider using the :func:`~mqt.core.dd.simulate` or the :func:`~mqt.core.dd.sample` functions, which never explicitly construct the state vector, for larger quantum computations.

Notes:
    This function internally constructs a :class:`~mqt.core.dd.DDPackage`, creates the zero state, and simulates the quantum computation via the :func:`simulate` function.
    The state vector is then extracted from the resulting DD via the :meth:`~mqt.core.dd.VectorDD.get_vector` method.

Args:
    qc: The quantum computation. Must only contain unitary operations.

Returns:
    The state vector of the final state.)pb");

  m.def(
      "build_unitary",
      [](const qc::QuantumComputation& qc, const bool recursive = false) {
        const auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        const auto u = recursive ? buildFunctionalityRecursive(qc, *dd)
                                 : buildFunctionality(qc, *dd);
        return getMatrix(u, qc.getNqubits());
      },
      "qc"_a, "recursive"_a = false,
      R"pb(Build a unitary matrix representation of a quantum computation.

This function builds a matrix representation of the unitary representing the functionality of a quantum computation.
This function does not support measurements, resets, or classical control, as the corresponding operations are non-unitary.

Since the unitary matrix is guaranteed to be exponentially large in the number of qubits, this function is only suitable for small quantum computations.
Consider using the :func:`~mqt.core.dd.build_functionality` function, which never explicitly constructs the unitary matrix, for larger quantum computations.

Notes:
    This function internally constructs a :class:`~mqt.core.dd.DDPackage`, creates the identity matrix, and builds the unitary matrix via the :func:`~mqt.core.dd.build_functionality` function.
    The unitary matrix is then extracted from the resulting DD via the :meth:`~mqt.core.dd.MatrixDD.get_matrix` method.

Args:
    qc: The quantum computation. Must only contain unitary operations.
    recursive: Whether to build the unitary matrix recursively.
        If set to True, the unitary matrix is built recursively by pairwise grouping the operations of the quantum computation.
        If set to False, the unitary matrix is built by sequentially applying the operations of the quantum computation to the identity matrix.
        Defaults to False.

Returns:
    The unitary matrix representing the functionality of the quantum computation.)pb");

  m.def("simulate", &dd::simulate, "qc"_a, "initial_state"_a, "dd_package"_a,
        R"pb(Simulate a quantum computation.

This function classically simulates a quantum computation for a given initial state and returns the final state (represented as a DD).
Compared to the `sample` function, this function does not support measurements, resets, or classical control.
It only supports unitary operations.

The simulation is effectively computed by sequentially applying the operations of the quantum computation to the initial state.

Args:
    qc: The quantum computation. Must only contain unitary operations.
    initial_state: The initial state as a DD. Must have the same number of qubits as the quantum computation.
        The reference count of the initial state is decremented during the simulation, so the caller must ensure that the initial state has a non-zero reference count.
    dd_package: The DD package. Must be configured with a sufficient number of qubits to accommodate the quantum computation.

Returns:
    The final state as a DD. The reference count of the final state is non-zero and must be manually decremented by the caller if it is no longer needed.)pb");

  m.def(
      "build_functionality",
      [](const qc::QuantumComputation& qc, dd::Package& p,
         const bool recursive = false) {
        if (recursive) {
          return buildFunctionalityRecursive(qc, p);
        }
        return buildFunctionality(qc, p);
      },
      "qc"_a, "dd_package"_a, "recursive"_a = false,
      R"pb(Build a functional representation of a quantum computation.

This function builds a matrix DD representation of the unitary representing the functionality of a quantum computation.
This function does not support measurements, resets, or classical control, as the corresponding operations are non-unitary.

Args:
    qc: The quantum computation.
        Must only contain unitary operations.
    dd_package: The DD package. Must be configured with a sufficient number of qubits to accommodate the quantum computation.
    recursive: Whether to build the functionality matrix recursively.
        If set to True, the functionality matrix is built recursively by pairwise grouping the operations of the quantum computation.
        If set to False, the functionality matrix is built by sequentially applying the operations of the quantum computation to the identity matrix.
        Defaults to False.

Returns:
    The functionality as a DD. The reference count of the result is non-zero and must be manually decremented by the caller if it is no longer needed.)pb");
}

} // namespace mqt
