// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_OQ3_IR_OQ3OPS_TD
#define MLIR_DIALECT_OQ3_IR_OQ3OPS_TD

include "mlir/Dialect/OQ3/IR/OQ3Attrs.td"
include "mlir/Dialect/OQ3/IR/OQ3Dialect.td"
include "mlir/Dialect/OQ3/IR/OQ3Types.td"
include "mlir/Dialect/QC/IR/QCTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class OQ3Op<string mnemonic, list<Trait> traits = []>
    : Op<OQ3Dialect, mnemonic, traits>;

def OQ3ScalarType
    : Type<
          Or<[AnyInteger.predicate, AnyFloat.predicate, AnyComplex.predicate,
              CPred<"::llvm::isa<::mlir::oq3::BitType, ::mlir::oq3::AngleType, "
                    "::mlir::oq3::DurationType, "
                    "::mlir::oq3::StretchType>($_self)">]>,
          "an OpenQASM scalar type">;

def GateOp : OQ3Op<"gate", [IsolatedFromAbove, Symbol, SingleBlock]> {
  let summary = "Typed OpenQASM gate definition";
  let arguments = (ins StrAttr:$sym_name, TypeAttr:$function_type);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name `:` $function_type $body attr-dict
  }];
  let hasVerifier = 1;
}

def GateDeclOp : OQ3Op<"gate_decl", [Symbol]> {
  let summary = "Typed declaration of an externally defined OpenQASM gate";
  let arguments = (ins StrAttr:$sym_name, TypeAttr:$function_type);
  let assemblyFormat = "$sym_name `:` $function_type attr-dict";
  let hasVerifier = 1;
}

def ApplyGateOp : OQ3Op<"apply_gate", [AttrSizedOperandSegments]> {
  let summary = "Apply a resolved OpenQASM gate with ordered modifiers";
  let arguments = (ins FlatSymbolRefAttr:$callee,
      Variadic<OQ3ScalarType>:$parameters, Variadic<QubitType>:$qubits,
      Variadic<OQ3ScalarType>:$modifier_operands,
      DenseI32ArrayAttr:$modifier_kinds,
      DenseI32ArrayAttr:$modifier_operand_indices);
  let assemblyFormat = [{
    $callee `(` $parameters `)` $qubits
    `modifiers` `[` $modifier_kinds `]` `(` $modifier_operands `)`
    `indices` `[` $modifier_operand_indices `]`
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def PackBitsOp : OQ3Op<"pack_bits", [Pure]> {
  let summary = "Pack ordered MLIR boolean values into an OpenQASM bit value";
  let arguments = (ins Variadic<I1>:$bits);
  let results = (outs BitType:$result);
  let assemblyFormat = "$bits attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def UnpackBitOp : OQ3Op<"unpack_bit", [Pure]> {
  let summary = "Extract one MLIR boolean from an OpenQASM bit value";
  let arguments = (ins BitType:$value, I64Attr:$index);
  let results = (outs I1:$result);
  let assemblyFormat = "$value `[` $index `]` attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def ForOp
    : OQ3Op<"for", [SingleBlock, AllTypesMatch<["start", "stop", "step"]>]> {
  let summary = "Inclusive OpenQASM integer range loop";
  let arguments = (ins AnyInteger:$start, AnyInteger:$stop, AnyInteger:$step);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $start `to` $stop `step` $step $body attr-dict `:` type($start)
  }];
  let hasVerifier = 1;
}

def YieldOp : OQ3Op<"yield", [Pure, Terminator]> {
  let summary = "Terminate an OQ3 region";
  let assemblyFormat = "attr-dict";
}

#endif // MLIR_DIALECT_OQ3_IR_OQ3OPS_TD
