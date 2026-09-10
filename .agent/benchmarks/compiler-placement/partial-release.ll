module {
  llvm.mlir.global internal constant @qir.result_label_c0("c0\00") {addr_space = 0 : i32, dso_local}
  llvm.func @main() -> i64 attributes {passthrough = ["entry_point", ["output_labeling_schema", "labeled"], ["qir_profiles", "adaptive_profile"], ["required_num_qubits", "0"], ["required_num_results", "0"]]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.addressof @qir.result_label_c0 : !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__quantum__rt__initialize(%6) : (!llvm.ptr) -> ()
    %7 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @__quantum__rt__result_array_allocate(%0, %7, %6) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %8 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @__quantum__rt__qubit_array_allocate(%0, %8, %6) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb2(%4 : i64)
  ^bb2(%9: i64):  // 2 preds: ^bb1, ^bb3
    %10 = llvm.icmp "slt" %9, %2 : i64
    llvm.cond_br %10, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %11 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__x__body(%11) : (!llvm.ptr) -> ()
    %12 = llvm.add %9, %3 : i64
    llvm.br ^bb2(%12 : i64)
  ^bb4:  // pred: ^bb2
    %13 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
    %14 = llvm.load %7 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__mz__body(%13, %14) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__qubit_release(%13) : (!llvm.ptr) -> ()
    llvm.call @__quantum__rt__qubit_array_release(%0, %8) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // pred: ^bb4
    llvm.call @__quantum__rt__result_array_record_output(%3, %7, %5) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_array_release(%0, %7) : (i64, !llvm.ptr) -> ()
    llvm.return %1 : i64
  }
  llvm.func @__quantum__rt__initialize(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__x__body(!llvm.ptr)
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) attributes {passthrough = ["irreversible"]}
  llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_array_release(i64, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_record_output(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_release(i64, !llvm.ptr)
  llvm.module_flags [#llvm.mlir.module_flag<error, "qir_major_version", 2 : i32>, #llvm.mlir.module_flag<max, "qir_minor_version", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_qubit_management", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_result_management", 1 : i32>, #llvm.mlir.module_flag<error, "backwards_branching", 1 : i32>, #llvm.mlir.module_flag<error, "arrays", 1 : i32>, #llvm.mlir.module_flag<append, "int_computations", ["i64"]>]
}
