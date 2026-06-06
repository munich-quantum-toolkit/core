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
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h" // NOLINT(misc-include-cleaner)
#include "qasm3/Importer.hpp"

#include <llvm/Support/raw_ostream.h>   // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinOps.h>         // NOLINT(misc-include-cleaner)
#include <mlir/IR/MLIRContext.h>        // NOLINT(misc-include-cleaner)
#include <mlir/IR/OwningOpRef.h>        // NOLINT(misc-include-cleaner)
#include <mlir/Support/LogicalResult.h> // NOLINT(misc-include-cleaner)

#include <stdexcept> // NOLINT(misc-include-cleaner)
#include <string>    // NOLINT(misc-include-cleaner)

namespace nb = nanobind; // NOLINT(misc-unused-alias-decls)

/// Build a fresh MLIRContext with all MQT dialects registered.
static mlir::MLIRContext makeContext() {
  mlir::MLIRContext ctx;
  MlirContext cCtx{&ctx};
  mqtRegisterAllDialects(cCtx);
  return ctx;
}

/// Parse a QASM string and translate it to a QC-dialect module.
/// Throws std::runtime_error on parse or translation failure.
static mlir::OwningOpRef<mlir::ModuleOp> importQasm(const std::string& qasm,
                                                     mlir::MLIRContext& ctx) {
  const auto qc = qasm3::Importer::imports(qasm);
  auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
  if (!module) {
    throw std::runtime_error(
        "failed to translate QASM circuit to QC dialect MLIR");
  }
  return module;
}

// NOLINTNEXTLINE(misc-use-internal-linkage,readability-identifier-naming,readability-named-parameter)
NB_MODULE(_mqtCoreMlir, m) {
  mqtRegisterAllPasses();

  m.doc() = "MQT Core MLIR Python bindings";

  // -------------------------------------------------------------------------
  // Dialect registration — exposed so MQTContext can delegate to C++.
  // -------------------------------------------------------------------------
  m.def(
      "_register_dialects",
      [](MlirContext ctx) { mqtRegisterAllDialects(ctx); }, nb::arg("context"),
      "Register all MQT MLIR dialects into the given context.");

  // -------------------------------------------------------------------------
  // Stage 1: QASM → QC dialect IR string.
  // -------------------------------------------------------------------------
  m.def(
      "load_qasm",
      [](const std::string& qasm) -> std::string {
        mlir::MLIRContext ctx = makeContext();
        auto module = importQasm(qasm, ctx);
        return mlir::captureIR(module.get());
      },
      nb::arg("qasm"),
      "Parse a QASM string and return the QC-dialect MLIR module as text.");

  // -------------------------------------------------------------------------
  // Full pipeline: QASM → compile → final IR (+ optional record).
  // -------------------------------------------------------------------------
  m.def(
      "compile_qasm",
      [](const std::string& qasm, bool convertToQIR,
         bool disableMergeSingleQubitRotationGates, bool enableHadamardLifting,
         bool captureIntermediates) -> nb::object {
        mlir::MLIRContext ctx = makeContext();
        auto module = importQasm(qasm, ctx);

        mlir::QuantumCompilerConfig cfg;
        cfg.convertToQIR = convertToQIR;
        cfg.disableMergeSingleQubitRotationGates =
            disableMergeSingleQubitRotationGates;
        cfg.enableHadamardLifting = enableHadamardLifting;
        cfg.recordIntermediates = captureIntermediates;

        mlir::CompilationRecord rec;
        if (mlir::failed(mlir::QuantumCompilerPipeline(cfg).runPipeline(
                module.get(), captureIntermediates ? &rec : nullptr))) {
          throw std::runtime_error("MQT compiler pipeline failed");
        }

        if (!captureIntermediates) {
          return nb::str(mlir::captureIR(module.get()).c_str());
        }

        // Return a dict whose keys mirror CompilationRecord field names
        // (snake_case) plus "result" for the final module.
        nb::dict stages;
        stages["result"] = nb::str(mlir::captureIR(module.get()).c_str());
        stages["after_qc_import"] = nb::str(rec.afterQCImport.c_str());
        stages["after_initial_canon"] = nb::str(rec.afterInitialCanon.c_str());
        stages["after_qco_conversion"] =
            nb::str(rec.afterQCOConversion.c_str());
        stages["after_qco_canon"] = nb::str(rec.afterQCOCanon.c_str());
        stages["after_optimization"] = nb::str(rec.afterOptimization.c_str());
        stages["after_optimization_canon"] =
            nb::str(rec.afterOptimizationCanon.c_str());
        stages["after_qc_conversion"] = nb::str(rec.afterQCConversion.c_str());
        stages["after_qc_canon"] = nb::str(rec.afterQCCanon.c_str());
        stages["after_qir_conversion"] =
            nb::str(rec.afterQIRConversion.c_str());
        stages["after_qir_canon"] = nb::str(rec.afterQIRCanon.c_str());
        return stages;
      },
      nb::arg("qasm"), nb::arg("convert_to_qir") = false,
      nb::arg("disable_merge_single_qubit_rotation_gates") = false,
      nb::arg("enable_hadamard_lifting") = false,
      nb::arg("capture_intermediates") = false,
      "Run the full MQT compiler pipeline on a QASM string.\n\n"
      "Returns the final IR string by default. Pass "
      "capture_intermediates=True\n"
      "to receive a dict of all stage snapshots plus a 'result' key.");
}
