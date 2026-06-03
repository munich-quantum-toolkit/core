/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h" // NOLINT(misc-include-cleaner)
#include "mlir/CAPI/Dialects.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h" // NOLINT(misc-include-cleaner)
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h" // NOLINT(misc-include-cleaner)
#include "mlir/Support/Passes.h" // NOLINT(misc-include-cleaner)
#include "qasm3/Importer.hpp"

#include <llvm/Support/raw_ostream.h>   // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinOps.h>         // NOLINT(misc-include-cleaner)
#include <mlir/IR/MLIRContext.h>        // NOLINT(misc-include-cleaner)
#include <mlir/Pass/PassManager.h>      // NOLINT(misc-include-cleaner)
#include <mlir/Support/LogicalResult.h> // NOLINT(misc-include-cleaner)

#include <stdexcept> // NOLINT(misc-include-cleaner)
#include <string>    // NOLINT(misc-include-cleaner)

namespace nb = nanobind; // NOLINT(misc-unused-alias-decls)

// NOLINTNEXTLINE(misc-use-internal-linkage,readability-identifier-naming,readability-named-parameter)
NB_MODULE(_mqtCoreMlir, m) {
  mqtMlirRegisterAllPasses();

  m.doc() = "MQT Core MLIR Python bindings";

  auto registerDialects = [](MlirContext context) {
    mqtMlirRegisterAllDialects(context);
  };
  m.def("register_dialects", registerDialects, nb::arg("context"),
        "Register and load QC, QCO, QTensor, and dependent MLIR dialects.");

  m.def(
      "qasm_to_qco",
      [](const std::string& qasm) -> std::string {
        auto qc = qasm3::Importer::imports(qasm);

        mlir::MLIRContext ctx;
        MlirContext cCtx{&ctx};
        mqtMlirRegisterAllDialects(cCtx);

        auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
        if (!module) {
          throw std::runtime_error("failed to translate circuit to QC MLIR");
        }

        mlir::PassManager pm(&ctx);
        populateQCCleanupPipeline(pm);
        pm.addPass(mlir::createQCToQCO());
        if (mlir::failed(pm.run(*module))) {
          throw std::runtime_error("qc-to-qco conversion failed");
        }

        std::string out;
        llvm::raw_string_ostream os(out);
        module->print(os);
        return out;
      },
      nb::arg("qasm"),
      "Run the full (py:qasm) -> (mlir:qc) -> (mlir:qco) pipeline.");
}
