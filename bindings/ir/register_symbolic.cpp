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
void registerVariable(const nb::module_& m);
void registerTerm(const nb::module_& m);
void registerExpression(const nb::module_& m);

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerSymbolic(const nb::module_& m) {
  registerVariable(m);
  registerTerm(m);
  registerExpression(m);
}
} // namespace mqt
