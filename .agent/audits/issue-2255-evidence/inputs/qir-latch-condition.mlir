module {
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__read_result__body(!llvm.ptr) -> i1
  llvm.func @main() attributes {passthrough = ["entry_point"]} {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %result = llvm.inttoptr %zero : i64 to !llvm.ptr
    llvm.br ^header
  ^header:
    llvm.br ^latch
  ^latch:
    llvm.call @__quantum__qis__mz__body(%result, %result)
        : (!llvm.ptr, !llvm.ptr) -> ()
    %again = llvm.call @__quantum__qis__read_result__body(%result)
        : (!llvm.ptr) -> i1
    llvm.cond_br %again, ^header, ^exit
  ^exit:
    llvm.return
  }
}
