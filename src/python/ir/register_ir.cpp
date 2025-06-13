/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)
// clang-format on

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// forward declarations
void registerRegisters(py::module& m);
void registerPermutation(py::module& m);
void registerOperations(py::module& m);
void registerSymbolic(py::module& m);
void registerQuantumComputation(py::module& m);

PYBIND11_MODULE(ir, m, py::mod_gil_not_used()) {
  registerPermutation(m);
  py::module registers = m.def_submodule("registers");
  registerRegisters(registers);

  py::module symbolic = m.def_submodule("symbolic");
  registerSymbolic(symbolic);

  py::module operations = m.def_submodule("operations");
  registerOperations(operations);

  registerQuantumComputation(m);
}

} // namespace mqt
