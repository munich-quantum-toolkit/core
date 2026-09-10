module {
  llvm.mlir.global private constant @__constant_8xf64(dense<[2.3561944901923448, 4.7123889803846897, 3.1415926535897931, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<8xf64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<8 x f64>
  llvm.mlir.global internal constant @qir.result_label_result("result\00") {addr_space = 0 : i32, dso_local}
  llvm.func @main() -> i64 attributes {passthrough = ["entry_point", ["output_labeling_schema", "labeled"], ["qir_profiles", "adaptive_profile"], ["required_num_qubits", "0"], ["required_num_results", "0"]]} {
    %0 = llvm.mlir.constant(8 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(7 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.addressof @__constant_8xf64 : !llvm.ptr
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.mlir.constant(-1.5707963267948966 : f64) : f64
    %8 = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %9 = llvm.mlir.addressof @qir.result_label_result : !llvm.ptr
    %10 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__quantum__rt__initialize(%10) : (!llvm.ptr) -> ()
    %11 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x f64>
    %12 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @__quantum__rt__result_array_allocate(%0, %12, %10) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %13 = llvm.call @__quantum__rt__qubit_allocate(%10) : (!llvm.ptr) -> !llvm.ptr
    %14 = llvm.call @__quantum__rt__qubit_allocate(%10) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @__quantum__qis__x__body(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb2(%3 : i64)
  ^bb2(%15: i64):  // 2 preds: ^bb1, ^bb8
    %16 = llvm.icmp "slt" %15, %6 : i64
    llvm.cond_br %16, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    %17 = llvm.sub %2, %15 : i64
    %18 = llvm.getelementptr inbounds|nuw %11[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %19 = llvm.load %18 : !llvm.ptr -> f64
    llvm.call @__quantum__qis__h__body(%13) : (!llvm.ptr) -> ()
    llvm.call @__quantum__qis__cp__body(%19, %13, %14) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %20 = llvm.sub %15, %5 : i64
    llvm.br ^bb4(%3, %7 : i64, f64)
  ^bb4(%21: i64, %22: f64):  // 2 preds: ^bb3, ^bb7
    %23 = llvm.icmp "slt" %21, %15 : i64
    llvm.cond_br %23, ^bb5, ^bb8
  ^bb5:  // pred: ^bb4
    %24 = llvm.sub %20, %21 : i64
    %25 = llvm.getelementptr %12[%24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %26 = llvm.load %25 : !llvm.ptr -> !llvm.ptr
    %27 = llvm.call @__quantum__rt__read_result(%26) : (!llvm.ptr) -> i1
    llvm.cond_br %27, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    llvm.call @__quantum__qis__p__body(%22, %13) : (f64, !llvm.ptr) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %28 = llvm.fmul %22, %8 : f64
    %29 = llvm.add %21, %5 : i64
    llvm.br ^bb4(%29, %28 : i64, f64)
  ^bb8:  // pred: ^bb4
    llvm.call @__quantum__qis__h__body(%13) : (!llvm.ptr) -> ()
    %30 = llvm.getelementptr %12[%15] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.load %30 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__mz__body(%13, %31) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__qis__reset__body(%13) : (!llvm.ptr) -> ()
    %32 = llvm.add %15, %5 : i64
    llvm.br ^bb2(%32 : i64)
  ^bb9:  // pred: ^bb2
    llvm.call @__quantum__rt__qubit_release(%13) : (!llvm.ptr) -> ()
    llvm.call @__quantum__rt__qubit_release(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // pred: ^bb9
    llvm.call @__quantum__rt__result_array_record_output(%6, %12, %9) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_array_release(%0, %12) : (i64, !llvm.ptr) -> ()
    llvm.return %1 : i64
  }
  llvm.func @__quantum__rt__initialize(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_allocate(!llvm.ptr) -> !llvm.ptr
  llvm.func @__quantum__rt__result_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__x__body(!llvm.ptr)
  llvm.func @__quantum__qis__h__body(!llvm.ptr)
  llvm.func @__quantum__qis__cp__body(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1
  llvm.func @__quantum__qis__p__body(f64, !llvm.ptr)
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) attributes {passthrough = ["irreversible"]}
  llvm.func @__quantum__qis__reset__body(!llvm.ptr) attributes {passthrough = ["irreversible"]}
  llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
  llvm.func @__quantum__rt__result_array_record_output(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_release(i64, !llvm.ptr)
  llvm.module_flags [#llvm.mlir.module_flag<error, "qir_major_version", 2 : i32>, #llvm.mlir.module_flag<max, "qir_minor_version", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_qubit_management", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_result_management", 1 : i32>, #llvm.mlir.module_flag<error, "backwards_branching", 1 : i32>, #llvm.mlir.module_flag<error, "arrays", 1 : i32>, #llvm.mlir.module_flag<append, "int_computations", ["i64"]>, #llvm.mlir.module_flag<append, "float_computations", ["double"]>]
}
