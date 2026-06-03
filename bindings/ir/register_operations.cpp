/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <nanobind/nanobind.h>

namespace mqt {

namespace nb = nanobind;

// forward declarations
void registerOptype(const nb::module_& m);
void registerControl(const nb::module_& m);
void registerOperation(const nb::module_& m);
void registerStandardOperation(const nb::module_& m);
void registerCompoundOperation(const nb::module_& m);
void registerNonUnitaryOperation(const nb::module_& m);
void registerSymbolicOperation(const nb::module_& m);
void registerIfElseOperation(const nb::module_& m);

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerOperations(const nb::module_& m) {
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
