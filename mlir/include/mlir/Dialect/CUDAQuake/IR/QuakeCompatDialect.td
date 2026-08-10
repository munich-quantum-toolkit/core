// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATDIALECT_TD
#define MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATDIALECT_TD

include "mlir/IR/OpBase.td"

def QuakeCompatDialect : Dialect {
  let name = "quake";
  let summary = "Compatibility projection of CUDA-Q's Quake dialect";
  let cppNamespace = "::mlir::cudaq_compat::quake";
  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATDIALECT_TD
