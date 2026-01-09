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
void registerRegisters(const nb::module_& m);
void registerPermutation(const nb::module_& m);
void registerOperations(const nb::module_& m);
void registerSymbolic(const nb::module_& m);
void registerQuantumComputation(const nb::module_& m);

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  registerPermutation(m);

  const nb::module_ symbolic = m.def_submodule("symbolic");
  registerSymbolic(symbolic);

  const nb::module_ registers = m.def_submodule("registers");
  registerRegisters(registers);

  const nb::module_ operations = m.def_submodule("operations");
  registerOperations(operations);

  registerQuantumComputation(m);
}

} // namespace mqt
