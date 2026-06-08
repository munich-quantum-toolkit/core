/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_MLIR_CAPI_DIALECTS_H
#define MQT_MLIR_CAPI_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// QC dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QC, qc);

//===----------------------------------------------------------------------===//
// QCO dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QCO, qco);

//===----------------------------------------------------------------------===//
// Bulk registration helpers
//===----------------------------------------------------------------------===//

/** Register and load all MQT dialects (QC, QCO, QTensor, Arith, Func,
 *  MemRef, SCF) into the given context. */
MLIR_CAPI_EXPORTED void mqtRegisterAllDialects(MlirContext ctx);

/** Register all MQT passes (QC transforms, QCO transforms, QC->QCO). */
MLIR_CAPI_EXPORTED void mqtRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // MQT_MLIR_CAPI_DIALECTS_H
