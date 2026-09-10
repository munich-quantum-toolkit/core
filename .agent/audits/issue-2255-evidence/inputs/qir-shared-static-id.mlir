module {
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
  llvm.func @main() attributes {passthrough = ["entry_point"]} {
    %id = llvm.mlir.constant(7 : i64) : i64
    %qubit = llvm.inttoptr %id : i64 to !llvm.ptr
    %result = llvm.inttoptr %id : i64 to !llvm.ptr
    %label = llvm.mlir.zero : !llvm.ptr
    llvm.call @__quantum__qis__mz__body(%qubit, %result)
        : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_record_output(%result, %label)
        : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}
