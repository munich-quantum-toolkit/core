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
#include "dd/DDpackageConfig.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"
#include "python/pybind11.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <pybind11/numpy.h>
#include <vector>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// forward declarations
void registerVectorDDs(const py::module& mod);
void registerMatrixDDs(const py::module& mod);
void registerDDPackage(const py::module& mod);

struct Vector {
  dd::CVec v;
};
Vector getVector(const dd::vEdge& v, dd::fp threshold = 0.);

struct Matrix {
  std::vector<std::complex<dd::fp>> data;
  size_t n;
};
Matrix getMatrix(const dd::mEdge& m, size_t numQubits, dd::fp threshold = 0.);

PYBIND11_MODULE(dd, mod, py::mod_gil_not_used()) {
  // Vector Decision Diagrams
  registerVectorDDs(mod);

  // Matrix Decision Diagrams
  registerMatrixDDs(mod);

  // DD Package
  registerDDPackage(mod);

  mod.def(
      "sample",
      [](const qc::QuantumComputation& qc, const size_t shots = 1024U,
         const size_t seed = 0U) { return dd::sample(qc, shots, seed); },
      "qc"_a, "shots"_a = 1024U, "seed"_a = 0U);

  mod.def(
      "simulate_statevector",
      [](const qc::QuantumComputation& qc) {
        auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        auto in = dd->makeZeroState(qc.getNqubits());
        const auto sim = dd::simulate(qc, in, *dd);
        return getVector(sim);
      },
      "qc"_a);

  mod.def(
      "build_unitary",
      [](const qc::QuantumComputation& qc, const bool recursive = false) {
        auto dd = std::make_unique<dd::Package>(qc.getNqubits());
        auto u = recursive ? dd::buildFunctionalityRecursive(qc, *dd)
                           : dd::buildFunctionality(qc, *dd);
        return getMatrix(u, qc.getNqubits());
      },
      "qc"_a, "recursive"_a = false);

  mod.def("simulate", &dd::simulate, "qc"_a, "initial_state"_a, "dd_package"_a);

  mod.def(
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
