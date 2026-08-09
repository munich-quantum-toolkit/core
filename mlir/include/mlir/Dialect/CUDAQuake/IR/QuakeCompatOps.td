// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATOPS_TD
#define MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATOPS_TD

include "mlir/Dialect/CUDAQuake/IR/CCCompatTypes.td"
include "mlir/Dialect/CUDAQuake/IR/QuakeCompatDialect.td"
include "mlir/Dialect/CUDAQuake/IR/QuakeCompatTypes.td"
include "mlir/IR/OpBase.td"

class QuakeCompatOp<string mnemonic, list<Trait> traits = []>
    : Op<QuakeCompatDialect, mnemonic, traits>;

def QuakeAllocaOp : QuakeCompatOp<"alloca"> {
  let summary = "Allocate a qubit reference or vector";
  let arguments = (ins Optional<AnySignlessInteger>:$size);
  let results = (outs AnyType:$reference);
  let assemblyFormat = [{
    qualified(type($reference)) (`[` $size^ `:` type($size) `]`)? attr-dict
  }];
}

def QuakeDeallocOp : QuakeCompatOp<"dealloc"> {
  let summary = "Deallocate a qubit reference or vector";
  let arguments = (ins AnyType:$reference);
  let assemblyFormat = "$reference `:` qualified(type($reference)) attr-dict";
}

def QuakeExtractRefOp : QuakeCompatOp<"extract_ref"> {
  let summary = "Extract a qubit reference from a vector";
  let arguments = (ins QuakeVeqType:$veq,
                       Optional<AnySignlessIntegerOrIndex>:$index,
                       I64Attr:$rawIndex);
  let results = (outs QuakeRefType:$reference);
  let assemblyFormat = [{
    $veq `[` custom<RawIndex>($index, $rawIndex) `]` `:`
      functional-type(operands, results) attr-dict
  }];
  let extraClassDeclaration = [{
    static constexpr uint64_t kDynamicIndex =
        std::numeric_limits<uint64_t>::max();
  }];
}

def QuakeRelaxSizeOp : QuakeCompatOp<"relax_size"> {
  let summary = "Forget the static size of a qubit vector";
  let arguments = (ins QuakeVeqType:$input);
  let results = (outs QuakeVeqType:$result);
  let assemblyFormat = "$input `:` functional-type(operands, results) attr-dict";
}

def QuakeConcatOp : QuakeCompatOp<"concat"> {
  let summary = "Concatenate qubit references and vectors";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs QuakeVeqType:$result);
  let assemblyFormat = "$inputs `:` functional-type(operands, results) attr-dict";
}

def QuakeApplyOp : QuakeCompatOp<"apply", [AttrSizedOperandSegments]> {
  let summary = "Apply a named CUDA-Q kernel";
  let arguments = (ins SymbolRefAttr:$callee,
                       UnitAttr:$is_adj,
                       Variadic<AnyType>:$controls,
                       Variadic<AnyType>:$actuals);
  let results = (outs Variadic<AnyType>:$results);
  let hasCustomAssemblyFormat = 1;
}

def QuakeResetOp : QuakeCompatOp<"reset"> {
  let summary = "Reset qubit references";
  let arguments = (ins Variadic<AnyType>:$targets);
  let results = (outs Variadic<AnyType>:$wires);
  let assemblyFormat = "$targets `:` functional-type(operands, results) attr-dict";
}

class QuakeMeasurement<string mnemonic> : QuakeCompatOp<mnemonic> {
  let arguments = (ins Variadic<AnyType>:$targets,
                       OptionalAttr<StrAttr>:$registerName);
  let results = (outs AnyType:$measurement, Variadic<AnyType>:$wires);
  let assemblyFormat = [{
    $targets (`name` $registerName^)? `:` functional-type(operands, results)
      attr-dict
  }];
}

def QuakeMxOp : QuakeMeasurement<"mx">;
def QuakeMyOp : QuakeMeasurement<"my">;
def QuakeMzOp : QuakeMeasurement<"mz">;

def QuakeDiscriminateOp : QuakeCompatOp<"discriminate"> {
  let summary = "Convert a measurement result to classical data";
  let arguments = (ins AnyType:$measurement);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$measurement `:` functional-type(operands, results) attr-dict";
}

class QuakeOperator<string mnemonic> :
    QuakeCompatOp<mnemonic, [AttrSizedOperandSegments]> {
  let arguments = (ins UnitAttr:$is_adj,
                       Variadic<AnyFloat>:$parameters,
                       Variadic<AnyType>:$controls,
                       Variadic<AnyType>:$targets,
                       OptionalAttr<DenseBoolArrayAttr>:$negated_qubit_controls);
  let results = (outs Variadic<AnyType>:$wires);
  let assemblyFormat = [{
    (`<` `adj` $is_adj^ `>`)? (`(` $parameters^ `)`)?
    (`[` $controls^ (`neg` $negated_qubit_controls^)? `]`)?
    $targets `:` functional-type(operands, results) attr-dict
  }];
}

def QuakeHOp : QuakeOperator<"h">;
def QuakePhasedRxOp : QuakeOperator<"phased_rx">;
def QuakeR1Op : QuakeOperator<"r1">;
def QuakeRxOp : QuakeOperator<"rx">;
def QuakeRyOp : QuakeOperator<"ry">;
def QuakeRzOp : QuakeOperator<"rz">;
def QuakeSOp : QuakeOperator<"s">;
def QuakeSwapOp : QuakeOperator<"swap">;
def QuakeTOp : QuakeOperator<"t">;
def QuakeU2Op : QuakeOperator<"u2">;
def QuakeU3Op : QuakeOperator<"u3">;
def QuakeXOp : QuakeOperator<"x">;
def QuakeYOp : QuakeOperator<"y">;
def QuakeZOp : QuakeOperator<"z">;

#endif // MLIR_DIALECT_CUDAQUAKE_IR_QUAKECOMPATOPS_TD
