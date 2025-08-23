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
void registerOptype(const py::module& m);
void registerControl(const py::module& m);
void registerOperation(const py::module& m);
void registerStandardOperation(const py::module& m);
void registerCompoundOperation(const py::module& m);
void registerNonUnitaryOperation(const py::module& m);
void registerSymbolicOperation(const py::module& m);
void registerIfElseOperation(const py::module& m);

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerOperations(py::module& m) {
  registerOptype(m);
  registerControl(m);
  registerOperation(m);
  registerStandardOperation(m);
  registerCompoundOperation(m);
  registerNonUnitaryOperation(m);
  registerSymbolicOperation(m);
  registerIfElseOperation(m);
}
} // namespace mqt
