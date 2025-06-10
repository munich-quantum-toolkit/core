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
void registerVariable(py::module& m);
void registerTerm(py::module& m);
void registerExpression(py::module& m);

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerSymbolic(pybind11::module& m) {
  registerVariable(m);
  registerTerm(m);
  registerExpression(m);
}

} // namespace mqt
