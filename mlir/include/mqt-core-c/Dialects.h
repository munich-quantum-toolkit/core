/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_C_DIALECTS_H
#define MQT_CORE_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QC, qc);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QCO, qco);

/**
 * @brief Register and load all dialects required by the MQT compilation
 * pipeline (QC, QCO, QTensor, and the upstream dialects they depend on).
 *
 * @param ctx The MLIR context to populate.
 */
MLIR_CAPI_EXPORTED void mqtRegisterDialects(MlirContext ctx);

/**
 * @brief Import an OpenQASM 3 program into a module of the QC dialect.
 *
 * @param ctx The MLIR context that owns the resulting module. The QC dialect
 * must be registered with this context.
 * @param qasm The OpenQASM 3 source program.
 * @return The resulting QC-dialect module, or a null module on error. The
 * caller owns the returned module.
 */
MLIR_CAPI_EXPORTED MlirModule mqtImportQASM3ToQC(MlirContext ctx,
                                                 MlirStringRef qasm);

/**
 * @brief Convert a QC-dialect module to the QCO dialect in place.
 *
 * @param module The module to convert. Modified in place on success.
 * @return @c true if the conversion succeeded, @c false otherwise.
 */
MLIR_CAPI_EXPORTED bool mqtConvertQCToQCO(MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // MQT_CORE_C_DIALECTS_H
