/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <vector>

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
      "qc"_a, "shots"_a = 1024U, "seed"_a = 0U);

  m.def(
      "simulate_statevector",
      [](const qc::QuantumComputation& qc) {
        auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        auto in = makeZeroState(qc.getNqubits(), *dd);
        const auto sim = dd::simulate(qc, in, *dd);
        return getVector(sim);
      },
      "qc"_a);

  m.def(
      "build_unitary",
      [](const qc::QuantumComputation& qc, const bool recursive = false) {
        auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        auto u = recursive ? dd::buildFunctionalityRecursive(qc, *dd)
                           : dd::buildFunctionality(qc, *dd);
        return getMatrix(u, qc.getNqubits());
      },
      "qc"_a, "recursive"_a = false);

  m.def("simulate", &dd::simulate, "qc"_a, "initial_state"_a, "dd_package"_a);

  m.def(
      "build_functionality",
      [](const qc::QuantumComputation& qc, dd::Package& p,
         const bool recursive = false) {
        if (recursive) {
          return dd::buildFunctionalityRecursive(qc, p);
        }
        return dd::buildFunctionality(qc, p);
      },
      "qc"_a, "dd_package"_a, "recursive"_a = false);
}

} // namespace mqt
