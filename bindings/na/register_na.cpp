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
void registerFomac(nb::module_& m);

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = R"pb(MQT Core NA - The MQT Core neutral atom module.

This module contains all neutral atom related functionality of MQT Core.)pb";

  nb::module_ fomac = m.def_submodule("fomac");
  registerFomac(fomac);
}

} // namespace mqt
