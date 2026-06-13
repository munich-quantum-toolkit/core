/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#ifndef MQT_CORE_C_REGISTRATION_H
#define MQT_CORE_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register and load all dialects of the MQT Compiler Collection with the
 * given context.
 *
 * @details Registers the QC, QCO, QTensor, and Jeff dialects together with the
 * upstream dialects (arith, func, scf, cf, memref, llvm) that the dialects,
 * conversions, and transformations depend on, and loads them so that modules
 * and passes can use them.
 *
 * @param ctx The MLIR context to populate.
 */
MLIR_CAPI_EXPORTED void mqtRegisterAllDialects(MlirContext ctx);

/**
 * @brief Register all conversion and transformation passes of the MQT Compiler
 * Collection with MLIR's global pass registry.
 *
 * @details After this call, every MQT pass can be instantiated by name (e.g.
 * via a textual pass pipeline) from Python through the standard MLIR
 * PassManager API.
 */
MLIR_CAPI_EXPORTED void mqtRegisterAllPasses(void);

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

#ifdef __cplusplus
}
#endif

#endif // MQT_CORE_C_REGISTRATION_H
