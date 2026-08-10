// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATTYPES_TD
#define MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATTYPES_TD

include "mlir/Dialect/CUDAQuake/IR/QuakeCompatDialect.td"
include "mlir/IR/AttrTypeBase.td"

class QuakeCompatType<string name, string typeMnemonic>
    : TypeDef<QuakeCompatDialect, name> {
  let mnemonic = typeMnemonic;
}

def QuakeRefType : QuakeCompatType<"Ref", "ref"> {
  let summary = "CUDA-Q reference-semantics qubit";
}

def QuakeVeqType : QuakeCompatType<"Veq", "veq"> {
  let summary = "CUDA-Q vector of qubit references";
  let parameters = (ins "uint64_t":$size);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static constexpr uint64_t kDynamicSize =
        std::numeric_limits<uint64_t>::max();

    [[nodiscard]] bool hasSpecifiedSize() const {
      return getSize() != kDynamicSize;
    }
  }];
}

def QuakeMeasureType : QuakeCompatType<"Measure", "measure"> {
  let summary = "CUDA-Q measurement result";
}

def QuakeWireType : QuakeCompatType<"Wire", "wire"> {
  let summary = "Unsupported CUDA-Q SSI qubit value";
}

def QuakeCableType : QuakeCompatType<"Cable", "cable"> {
  let summary = "Unsupported CUDA-Q SSI qubit vector";
  let parameters = (ins "uint64_t":$size);
  let assemblyFormat = "`<` $size `>`";
}

#endif // MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATTYPES_TD
