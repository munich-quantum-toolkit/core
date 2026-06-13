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

#include "ir/QuantumComputation.hpp"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Transforms/Passes.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/Transforms/Passes.h"
#include "qasm3/Importer.hpp"

#include <jeff/IR/JeffDialect.h>
#include <mlir/CAPI/IR.h>
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
#include <mlir/Transforms/Passes.h>

#include <exception>
#include <string>

void mqtRegisterAllDialects(MlirContext ctx) {
  mlir::MLIRContext* context = unwrap(ctx);
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::jeff::JeffDialect,
                  mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

void mqtRegisterAllPasses() {
  // Common upstream transformations (canonicalization, CSE, ...) used by the
  // MQT cleanup pipelines.
  mlir::registerTransformsPasses();

  // Conversions between the MQT dialects.
  mlir::registerQCToQCOPasses();
  mlir::registerQCOToQCPasses();
  mlir::registerQCToQIRBasePasses();
  mlir::registerQCToQIRAdaptivePasses();
  mlir::registerJeffToQCOPasses();
  mlir::registerQCOToJeffPasses();

  // Dialect-specific transformations.
  mlir::qc::registerQCPasses();
  mlir::qco::registerQCOPasses();
  mlir::qir::registerQIRPasses();
  mlir::qtensor::registerQTensorPasses();
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
