/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt-core-c/Dialects.h"

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h>
#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

namespace nb = nanobind;

NB_MODULE(_mqtCore, m) {
  m.doc() = "MQT Core MLIR dialects and compilation pipeline bindings";

  m.def(
      "register_dialects",
      [](MlirContext context) { mqtRegisterDialects(context); },
      nb::arg("context"),
      "Register and load all dialects used by the MQT compilation pipeline.");

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

  m.def(
      "qc_to_qco",
      [](MlirModule module) {
        if (!mqtConvertQCToQCO(module)) {
          throw std::runtime_error(
              "Failed to transform the QC-dialect module to the QCO dialect");
        }
        return module;
      },
      nb::arg("module"),
      "Transform a QC-dialect module to the QCO dialect in place.");
}
