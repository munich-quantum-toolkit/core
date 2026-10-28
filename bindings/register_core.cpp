/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "nanobind/nanobind.h"

namespace mqt {
void registerBench(nanobind::module_& mod);
void registerDD(const nanobind::module_& mod);
void registerQDMI(nanobind::module_& mod);
void registerMLIR(nanobind::module_& mod);
} // namespace mqt

NB_MODULE(_core, mod) {
  auto bench = mod.def_submodule("bench");
  const auto dd = mod.def_submodule("dd");
  auto qdmi = mod.def_submodule("qdmi");
  auto mlir = mod.def_submodule("mlir");
  // Set the public scope before nanobind records class and function module
  // names.
  bench.attr("__name__") = "mqt.core.bench";
  dd.attr("__name__") = "mqt.core.dd";
  qdmi.attr("__name__") = "mqt.core.qdmi";
  mlir.attr("__name__") = "mqt.core.mlir";
  mqt::registerBench(bench);
  mqt::registerDD(dd);
  mqt::registerQDMI(qdmi);
  mqt::registerMLIR(mlir);
}
