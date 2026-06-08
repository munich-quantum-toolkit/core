/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/CAPI/Dialects.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Transforms/Passes.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QC, qc, ::mlir::qc::QCDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QCO, qco, ::mlir::qco::QCODialect)

void mqtRegisterAllDialects(MlirContext ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  unwrap(ctx)->appendDialectRegistry(registry);
  unwrap(ctx)->loadAllAvailableDialects();
}

void mqtRegisterAllPasses() {
  mlir::qc::registerQCPasses();
  mlir::qco::registerQCOPasses();
  mlir::registerQCToQCOPasses();
}
