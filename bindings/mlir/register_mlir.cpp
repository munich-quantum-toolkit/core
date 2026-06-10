/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp" // NOLINT(misc-include-cleaner)
#include "mlir-c/Dialects.h"
#include "mlir/CAPI/IR.h" // NOLINT(misc-include-cleaner)
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h" // NOLINT(misc-include-cleaner)
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h" // NOLINT(misc-include-cleaner)
#include "mlir/Support/Passes.h" // NOLINT(misc-include-cleaner)
#include "qasm3/Importer.hpp"

#include <llvm/Support/raw_ostream.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinOps.h>       // NOLINT(misc-include-cleaner)
#include <mlir/IR/MLIRContext.h>      // NOLINT(misc-include-cleaner)
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>      // NOLINT(misc-include-cleaner)
#include <mlir/Support/LogicalResult.h> // NOLINT(misc-include-cleaner)
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

void setupContext(mlir::MLIRContext& ctx) {
  // NOLINTNEXTLINE(misc-include-cleaner)
  const MlirContext cCtx{&ctx};
  mqtMlirRegisterAllDialects(cCtx);
}

std::string moduleToString(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  mqtMlirRegisterAllPasses();

  m.doc() = "MQT Core MLIR Python bindings";

  m.def(
      "load_qasm",
      [](const std::string& qasm) -> std::string {
        auto qc = qasm3::Importer::imports(qasm);
        mlir::MLIRContext ctx;
        setupContext(ctx);
        auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
        if (!module) {
          throw std::runtime_error("failed to translate QASM to QC MLIR");
        }
        return moduleToString(*module);
      },
      nb::arg("qasm"),
      "Parse an OpenQASM string and return the QC dialect MLIR module as "
      "text.");

  m.def(
      "convert_qc_to_qco",
      [](const std::string& mlirText) -> std::string {
        mlir::MLIRContext ctx;
        setupContext(ctx);
        auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &ctx);
        if (!module) {
          throw std::runtime_error("failed to parse QC MLIR module");
        }
        mlir::PassManager pm(&ctx);
        populateQCCleanupPipeline(pm);
        pm.addPass(mlir::createQCToQCO());
        if (mlir::failed(pm.run(*module))) {
          throw std::runtime_error("qc-to-qco conversion failed");
        }
        return moduleToString(*module);
      },
      nb::arg("mlir_text"),
      "Convert a QC dialect module (as text) to QCO dialect.");

  m.def(
      "compile_program",
      [](const std::string& qasm, bool convertToQir) -> std::string {
        auto qc = qasm3::Importer::imports(qasm);
        mlir::MLIRContext ctx;
        setupContext(ctx);
        auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
        if (!module) {
          throw std::runtime_error("failed to translate QASM to QC MLIR");
        }
        const mlir::QuantumCompilerConfig config{.convertToQIR = convertToQir};
        const mlir::QuantumCompilerPipeline pipeline(config);
        if (mlir::failed(pipeline.runPipeline(*module))) {
          throw std::runtime_error("compilation pipeline failed");
        }
        return moduleToString(*module);
      },
      nb::arg("qasm"), nb::arg("convert_to_qir") = false,
      "Run the full compiler pipeline on an OpenQASM program.");
}
