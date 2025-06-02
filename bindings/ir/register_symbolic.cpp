/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "python/pybind11.hpp"

namespace mqt {

// forward declarations
void registerVariable(py::module& m);
void registerTerm(py::module& m);
void registerExpression(py::module& m);

void registerSymbolic(pybind11::module& m) {
  registerVariable(m);
  registerTerm(m);
  registerExpression(m);
}
} // namespace mqt
