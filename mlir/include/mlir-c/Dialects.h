/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_MLIR_C_DIALECTS_H
#define MQT_MLIR_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QC, qc);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QCO, qco);

/** Registers and loads all MQT MLIR dialects into a context. */
MLIR_CAPI_EXPORTED void mqtMlirRegisterAllDialects(MlirContext context);

/** Registers all MQT MLIR passes into the global registry. */
MLIR_CAPI_EXPORTED void mqtMlirRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // MQT_MLIR_C_DIALECTS_H
