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
void registerQdmi(nb::module_& m);

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = R"pb(MQT Core NA - The MQT Core Neutral Atom module.

This module contains all neutral atom related functionality of MQT Core.)pb";

  nb::module_ qdmi = m.def_submodule("qdmi");
  registerQdmi(qdmi);

  const nb::module_ fomac = m.def_submodule("fomac");
  fomac.doc() = R"pb(Deprecated compatibility aliases for mqt.core.na.qdmi.

This module will be removed in MQT Core 4.0.)pb";
  fomac.attr("Device") = qdmi.attr("Device");
  fomac.attr("devices") = qdmi.attr("devices");
}

} // namespace mqt
