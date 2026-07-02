/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt-core-c/Registration.h"

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h>
#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

namespace mqt {

namespace nb = nanobind;

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() =
      R"pb(MQT Core MLIR - Python bindings for the MQT Compiler Collection.

This module exposes the dialects and passes of the MQT Compiler Collection so
that quantum programs can be compiled from Python using MLIR's Python bindings.)pb";

  m.def(
      "register_dialects",
      [](MlirContext context) { mqtRegisterAllDialects(context); },
      nb::arg("context"),
      "Register and load all MQT Compiler Collection dialects with the given "
      "context.");

  m.def(
      "register_passes", []() { mqtRegisterAllPasses(); },
      "Register all MQT Compiler Collection passes with MLIR's global pass "
      "registry.");

  m.def(
      "import_qasm3_to_qc",
      [](MlirContext context, const std::string& qasm) {
        const MlirModule module = mqtImportQASM3ToQC(
            context, mlirStringRefCreate(qasm.data(), qasm.size()));
        if (mlirModuleIsNull(module)) {
          throw std::runtime_error(
              "Failed to import OpenQASM 3 program into the QC dialect");
        }
        return module;
      },
      nb::arg("context"), nb::arg("qasm"),
      "Import an OpenQASM 3 program into a QC-dialect module.");
}

} // namespace mqt
