/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Bindings.hpp"

#include "nanobind/nanobind.h"

NB_MODULE(_core, module) {
  auto bench = module.def_submodule("bench");
  const auto dd = module.def_submodule("dd");
  auto qdmi = module.def_submodule("qdmi");
  auto mlir = module.def_submodule("mlir");
  // Keep public type names independent of the native extension's filename.
  bench.attr("__name__") = "mqt.core.bench";
  dd.attr("__name__") = "mqt.core.dd";
  qdmi.attr("__name__") = "mqt.core.qdmi";
  mlir.attr("__name__") = "mqt.core.mlir";
  mqt::registerBench(bench);
  mqt::registerDD(dd);
  mqt::registerQDMI(qdmi);
  mqt::registerMLIR(mlir);
}
