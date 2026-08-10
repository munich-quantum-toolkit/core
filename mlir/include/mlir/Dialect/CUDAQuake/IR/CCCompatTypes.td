// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATTYPES_TD
#define MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATTYPES_TD

include "mlir/Dialect/CUDAQuake/IR/CCCompatDialect.td"
include "mlir/IR/AttrTypeBase.td"

class CCCompatType<string name, string typeMnemonic>
    : TypeDef<CCCompatDialect, name> {
  let mnemonic = typeMnemonic;
}

def CCCallableType : CCCompatType<"Callable", "callable"> {
  let summary = "CUDA-Q callable value";
  let parameters = (ins "::mlir::FunctionType":$signature);
  let assemblyFormat = "`<` $signature `>`";
}

def CCStdvecType : CCCompatType<"Stdvec", "stdvec"> {
  let summary = "CUDA-Q vector-like classical value";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
}

def CCMeasureHandleType : CCCompatType<"MeasureHandle", "measure_handle"> {
  let summary = "CUDA-Q opaque measurement handle";
}

#endif // MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATTYPES_TD
