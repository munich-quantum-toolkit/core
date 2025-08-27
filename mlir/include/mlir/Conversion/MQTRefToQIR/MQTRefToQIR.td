// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTRefToQIR : Pass<"mqtref-to-qir"> {
   let summary = "Convert MQT's `Ref` dialect into the llvm dialect that adheres to the QIR specification";

   let description = [{
      This pass converts all operations to the llvm dialect. The quantum operations of the mqtref dialect are converted
      to an equivalent function in the QIR instruction set with a function call to it in the llvm dialect. This requires a valid
      input in the mqtref dialect either using static or dynamically addressed qubits where the main function is marked with the
      entry_point attribute. At the moment the pass only supports simple quantum programs with a straight controlflow and converts
      them to the llvm dialect according the QIR base profile. The required module flags are added as additional attributes to the main function.
      The initial main function is divided into 4 blocks as required of the base profile. The first block only contains operations that
      are required to initialize the execution environment. The second and third block only contains calls to quantum operations that return void.
      The difference between these two blocks is that operations in the second block are reversible while the call operations
      in the third block are to functions that are marked as irreversible. These are at the moment `__quantum__qis__mz__body`,
      `__quantum__rt__qubit_release_array`, `__quantum__rt__qubit_release`, and `__quantum__qis__reset__body`.
      The last block only contains operations to record the output of the measurements and the return operation. The individual blocks
      are connected with unconditional jump operations to the next block.
      The `mqtref.measure` operation is converted into a `__quantum__qis__mz__body` operation that performs the measurement and a
      `__quantum__rt__result_record_output` operation that records the measurement outputs.
      The remaining operations from the standard dialects are lowered to the llvm dialect via built-in conversion passes from MLIR.
      This includes the following dialects at the moment: func, arith, cf

      Once the module is converted fully in the llvm dialect, one can convert the resulting file into LLVM-IR with the following command:
      mlir-translate --mlir-to-llvmir input.mlir > output.ll
   }];

  // Define dependent dialects
  let dependentDialects = [
     "::mqt::ir::ref::MQTRefDialect",
     "mlir::LLVM::LLVMDialect"
  ];
}
