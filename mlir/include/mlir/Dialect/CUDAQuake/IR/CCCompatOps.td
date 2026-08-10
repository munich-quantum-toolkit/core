// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATOPS_TD
#define MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATOPS_TD

include "mlir/Dialect/CUDAQuake/IR/CCCompatDialect.td"
include "mlir/Dialect/CUDAQuake/IR/CCCompatTypes.td"
include "mlir/IR/OpBase.td"

class CCCompatOp<string mnemonic, list<Trait> traits = []>
    : Op<CCCompatDialect, mnemonic, traits>;

def CCScopeOp : CCCompatOp<"scope"> {
  let summary = "CUDA-Q lexical scope";
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

def CCCreateLambdaOp : CCCompatOp<"create_lambda"> {
  let summary = "Create a CUDA-Q callable lambda";
  let results = (outs CCCallableType:$callable);
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

def CCReturnOp : CCCompatOp<"return", [Terminator]> {
  let summary = "Return from a CUDA-Q callable lambda";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = [{
    ($values^ `:` qualified(type($values)))? attr-dict
  }];
}

def CCIfOp : CCCompatOp<"if"> {
  let summary = "CUDA-Q structured if operation";
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
  let hasCustomAssemblyFormat = 1;
}

def CCLoopOp : CCCompatOp<"loop"> {
  let summary = "CUDA-Q generalized structured loop";
  let arguments = (ins Variadic<AnyType>:$initialArgs,
      BoolAttr:$post_condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$whileRegion, AnyRegion:$bodyRegion,
      AnyRegion:$stepRegion, AnyRegion:$elseRegion);
  let hasCustomAssemblyFormat = 1;
}

def CCConditionOp : CCCompatOp<"condition", [Terminator]> {
  let summary = "CUDA-Q loop condition terminator";
  let arguments = (ins I1:$condition, Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $condition (`(` $results^ `:` qualified(type($results)) `)`)? attr-dict
  }];
}

def CCContinueOp : CCCompatOp<"continue", [Terminator]> {
  let summary = "CUDA-Q structured-region continuation";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = [{
    ($values^ `:` qualified(type($values)))? attr-dict
  }];
}

def CCBreakOp : CCCompatOp<"break", [Terminator]> {
  let summary = "CUDA-Q loop break terminator";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = [{
    ($values^ `:` qualified(type($values)))? attr-dict
  }];
}

#endif // MLIR_DIALECT_CUDAQUAKE_IR_CCCOMPATOPS_TD
