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
#include "ir/QuantumComputation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

#include <complex>
#include <cstddef>
#include <memory>

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
  m.doc() = R"pb(MQT Core DD  - The MQT Core Decision Diagram (DD) module.)pb";

  nb::module_::import_("mqt.core.ir");

  // Vector Decision Diagrams
  registerVectorDDs(m);

  // Matrix Decision Diagrams
  registerMatrixDDs(m);

  // DD Package
  registerDDPackage(m);

  m.def(
      "build_unitary",
      [](const qc::QuantumComputation& qc) {
        const auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        const auto u = buildFunctionality(qc, *dd);
        return getMatrix(u, qc.getNqubits());
      },
      "qc"_a,
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

Returns:
    The unitary matrix representing the functionality of the quantum computation.)pb");

  m.def("simulate", &dd::simulate, "qc"_a, "initial_state"_a, "dd_package"_a,
        R"pb(Simulate a quantum computation.

This function classically simulates a quantum computation for a given initial state and returns the final state (represented as a DD).
This function only supports unitary operations; it does not support measurements, resets, or classical control.

The simulation is effectively computed by sequentially applying the operations of the quantum computation to the initial state.

Args:
    qc: The quantum computation. Must only contain unitary operations.
    initial_state: The initial state as a DD. Must have the same number of qubits as the quantum computation.
        The reference count of the initial state is decremented during the simulation, so the caller must ensure that the initial state has a non-zero reference count.
    dd_package: The DD package. Must be configured with a sufficient number of qubits to accommodate the quantum computation.

Returns:
    The final state as a DD. The reference count of the final state is non-zero and must be manually decremented by the caller if it is no longer needed.)pb");

  m.def("build_functionality", &dd::buildFunctionality, "qc"_a, "dd_package"_a,
        nb::keep_alive<0, 2>(),
        R"pb(Build a functional representation of a quantum computation.

This function builds a matrix DD representation of the unitary representing the functionality of a quantum computation.
This function does not support measurements, resets, or classical control, as the corresponding operations are non-unitary.

Args:
    qc: The quantum computation.
        Must only contain unitary operations.
    dd_package: The DD package. Must be configured with a sufficient number of qubits to accommodate the quantum computation.

Returns:
    The functionality as a DD. The reference count of the result is non-zero and must be manually decremented by the caller if it is no longer needed.)pb");
}

} // namespace mqt
