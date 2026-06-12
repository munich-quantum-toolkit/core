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

#include "ir/QuantumComputation.hpp"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "qasm3/Importer.hpp"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <exception>
#include <string>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QC, qc, mlir::qc::QCDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QCO, qco, mlir::qco::QCODialect)

void mqtRegisterDialects(MlirContext ctx) {
  mlir::MLIRContext* context = unwrap(ctx);
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

MlirModule mqtImportQASM3ToQC(MlirContext ctx, MlirStringRef qasm) {
  mlir::MLIRContext* context = unwrap(ctx);
  try {
    const ::qc::QuantumComputation qc =
        qasm3::Importer::imports(std::string(qasm.data, qasm.length));
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::translateQuantumComputationToQC(context, qc);
    if (!module) {
      return MlirModule{nullptr};
    }
    return wrap(module.release());
  } catch (const std::exception&) {
    return MlirModule{nullptr};
  }
}

bool mqtConvertQCToQCO(MlirModule module) {
  mlir::ModuleOp moduleOp = unwrap(module);
  mlir::PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::createQCToQCO());
  return mlir::succeeded(pm.run(moduleOp));
}
