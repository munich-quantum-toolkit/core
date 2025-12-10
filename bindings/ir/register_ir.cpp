/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
void registerRegisters(nb::module_& m);
void registerPermutation(nb::module_& m);
void registerOperations(nb::module_& m);
void registerSymbolic(nb::module_& m);
void registerQuantumComputation(nb::module_& m);

// NOLINTNEXTLINE
NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  registerPermutation(m);

  nb::module_ symbolic = m.def_submodule("symbolic");
  registerSymbolic(symbolic);

  nb::module_ registers = m.def_submodule("registers");
  registerRegisters(registers);

  nb::module_ operations = m.def_submodule("operations");
  registerOperations(operations);

  registerQuantumComputation(m);
}

} // namespace mqt
