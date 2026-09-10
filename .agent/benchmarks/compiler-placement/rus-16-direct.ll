module {
  llvm.mlir.global internal constant @qir.result_label_result("result\00") {addr_space = 0 : i32, dso_local}
  llvm.func @main() -> i64 attributes {passthrough = ["entry_point", ["output_labeling_schema", "labeled"], ["qir_profiles", "adaptive_profile"], ["required_num_qubits", "0"], ["required_num_results", "0"]]} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(16 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(0.78539816339744828 : f64) : f64
    %7 = llvm.mlir.constant(-3.1415926535897931 : f64) : f64
    %8 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %9 = llvm.mlir.constant(-2.3561944901923448 : f64) : f64
    %10 = llvm.mlir.addressof @qir.result_label_result : !llvm.ptr
    %11 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__quantum__rt__initialize(%11) : (!llvm.ptr) -> ()
    %12 = llvm.alloca %1 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @__quantum__rt__result_array_allocate(%1, %12, %11) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    %13 = llvm.call @__quantum__rt__result_allocate(%11) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %14 = llvm.call @__quantum__rt__qubit_allocate(%11) : (!llvm.ptr) -> !llvm.ptr
    %15 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @__quantum__rt__qubit_array_allocate(%0, %15, %11) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb9
    llvm.call @__quantum__qis__u2__body(%6, %7, %14) : (f64, f64, !llvm.ptr) -> ()
    llvm.br ^bb3(%3 : i64)
  ^bb3(%16: i64):  // 2 preds: ^bb2, ^bb4
    %17 = llvm.icmp "slt" %16, %4 : i64
    llvm.cond_br %17, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %18 = llvm.getelementptr %15[%16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %19 = llvm.load %18 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__cx__body(%14, %19) : (!llvm.ptr, !llvm.ptr) -> ()
    %20 = llvm.add %16, %5 : i64
    llvm.br ^bb3(%20 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @__quantum__qis__h__body(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb6(%3 : i64)
  ^bb6(%21: i64):  // 2 preds: ^bb5, ^bb7
    %22 = llvm.icmp "slt" %21, %4 : i64
    llvm.cond_br %22, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %23 = llvm.getelementptr %15[%21] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %24 = llvm.load %23 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__cx__body(%14, %24) : (!llvm.ptr, !llvm.ptr) -> ()
    %25 = llvm.add %21, %5 : i64
    llvm.br ^bb6(%25 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @__quantum__qis__u2__body(%8, %9, %14) : (f64, f64, !llvm.ptr) -> ()
    llvm.call @__quantum__qis__mz__body(%14, %13) : (!llvm.ptr, !llvm.ptr) -> ()
    %26 = llvm.call @__quantum__rt__read_result(%13) : (!llvm.ptr) -> i1
    llvm.cond_br %26, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.call @__quantum__qis__x__body(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb10:  // pred: ^bb8
    %27 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__s__adj(%27) : (!llvm.ptr) -> ()
    llvm.br ^bb11(%3 : i64)
  ^bb11(%28: i64):  // 2 preds: ^bb10, ^bb12
    %29 = llvm.icmp "slt" %28, %4 : i64
    llvm.cond_br %29, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %30 = llvm.getelementptr %15[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.load %30 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__h__body(%31) : (!llvm.ptr) -> ()
    %32 = llvm.add %28, %5 : i64
    llvm.br ^bb11(%32 : i64)
  ^bb13:  // pred: ^bb11
    llvm.br ^bb14(%5 : i64)
  ^bb14(%33: i64):  // 2 preds: ^bb13, ^bb15
    %34 = llvm.icmp "slt" %33, %4 : i64
    llvm.cond_br %34, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %35 = llvm.getelementptr %15[%33] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %36 = llvm.load %35 : !llvm.ptr -> !llvm.ptr
    %37 = llvm.load %15 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__cx__body(%36, %37) : (!llvm.ptr, !llvm.ptr) -> ()
    %38 = llvm.add %33, %5 : i64
    llvm.br ^bb14(%38 : i64)
  ^bb16:  // pred: ^bb14
    %39 = llvm.load %12 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__mz__body(%27, %39) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__qubit_release(%14) : (!llvm.ptr) -> ()
    llvm.call @__quantum__rt__qubit_array_release(%0, %15) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb17
  ^bb17:  // pred: ^bb16
    llvm.call @__quantum__rt__result_array_record_output(%5, %12, %10) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_release(%13) : (!llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_array_release(%1, %12) : (i64, !llvm.ptr) -> ()
    llvm.return %2 : i64
  }
  llvm.func @__quantum__rt__initialize(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_allocate(!llvm.ptr) -> !llvm.ptr
  llvm.func @__quantum__rt__qubit_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__u2__body(f64, f64, !llvm.ptr)
  llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__h__body(!llvm.ptr)
  llvm.func @__quantum__rt__result_allocate(!llvm.ptr) -> !llvm.ptr
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) attributes {passthrough = ["irreversible"]}
  llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1
  llvm.func @__quantum__qis__x__body(!llvm.ptr)
  llvm.func @__quantum__qis__s__adj(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_array_release(i64, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_record_output(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_release(!llvm.ptr)
  llvm.func @__quantum__rt__result_array_release(i64, !llvm.ptr)
  llvm.module_flags [#llvm.mlir.module_flag<error, "qir_major_version", 2 : i32>, #llvm.mlir.module_flag<max, "qir_minor_version", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_qubit_management", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_result_management", 1 : i32>, #llvm.mlir.module_flag<error, "backwards_branching", 3 : i32>, #llvm.mlir.module_flag<error, "arrays", 1 : i32>, #llvm.mlir.module_flag<append, "int_computations", ["i64"]>]
}
