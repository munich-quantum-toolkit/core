module {
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__read_result__body(!llvm.ptr) -> i1
  llvm.func @main() attributes {passthrough = ["entry_point"]} {
    %qubit_id = llvm.mlir.constant(0 : i64) : i64
    %result_id = llvm.mlir.constant(7 : i64) : i64
    %qubit = llvm.inttoptr %qubit_id : i64 to !llvm.ptr
    %result = llvm.inttoptr %result_id : i64 to !llvm.ptr
    llvm.call @__quantum__qis__mz__body(%qubit, %result)
        : (!llvm.ptr, !llvm.ptr) -> ()
    %bit = llvm.call @__quantum__qis__read_result__body(%result)
        : (!llvm.ptr) -> i1
    llvm.return
  }
}
