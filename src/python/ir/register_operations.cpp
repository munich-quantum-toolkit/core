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
void registerOptype(const py::module& m);
void registerControl(const py::module& m);
void registerOperation(const py::module& m);
void registerStandardOperation(const py::module& m);
void registerCompoundOperation(const py::module& m);
void registerNonUnitaryOperation(const py::module& m);
void registerSymbolicOperation(const py::module& m);
void registerClassicControlledOperation(const py::module& m);

void registerOperations(py::module& m) {
  registerOptype(m);
  registerControl(m);
  registerOperation(m);
  registerStandardOperation(m);
  registerCompoundOperation(m);
  registerNonUnitaryOperation(m);
  registerSymbolicOperation(m);
  registerClassicControlledOperation(m);
}
} // namespace mqt
