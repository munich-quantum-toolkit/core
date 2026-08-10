// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATDIALECT_TD
#define MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATDIALECT_TD

include "mlir/IR/OpBase.td"

def CCCompatDialect : Dialect {
  let name = "cc";
  let summary = "Compatibility projection of CUDA-Q's classical dialect";
  let cppNamespace = "::mlir::cudaq_compat::cc";
  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATDIALECT_TD
