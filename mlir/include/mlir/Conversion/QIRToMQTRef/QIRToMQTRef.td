// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QIRToMQTRef : Pass<"qir-to-mqtref"> {
    let summary = "Convert llvm dialect that adheres to the QIR specification to MQT's `Ref` dialect";

    let description = [{
      This pass converts all llvm.mlir.call operations calling functions of QIR instruction set into quantum operations in the mqtref dialect.
      The pass expects a valid mlir module in the llvm dialect that uses either statically or dynamically addressed qubits. At the moment
      the pass only supports simple quantum programs with a straight control flow that follows the QIR base profile. The conversion of the quantum
      operation is done by checking the function name of the call operation and replacing it with an equivalent operation
      from the mqtref dialect. The `__quantum__qis__mz__body` operation is converted to a `mqtref.measure` operation that returns the
      result of the measurement as i1 value. The accompanying `__quantum__rt__result_record_output` operation of the measurement is erased
      during the conversion.
      At the moment llvm.mlir.constants and function declarations are currently not converted and persist after the conversion.
      Quantum operations that do not have an equivalent operation in the mqtref dialect or are not supported yet
      cause the conversion to fail.

      The initial .mlir module with the llvm dialect can be obtained by converting a .ll file with the following command:
      mlir-translate --import-llvm input.ll -o output.mlir
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::ref::MQTRefDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}
