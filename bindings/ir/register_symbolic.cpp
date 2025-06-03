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
<<<<<<< HEAD
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)
=======
#include <pybind11/stl.h>
>>>>>>> 59e09070 (Remove pybind11.hpp)
// clang-format on

namespace mqt {

namespace py = pybind11;
<<<<<<< HEAD
using namespace pybind11::literals;
=======
using namespace py::literals;
>>>>>>> 59e09070 (Remove pybind11.hpp)

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
