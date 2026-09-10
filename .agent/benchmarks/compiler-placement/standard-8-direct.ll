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
    %13 = llvm.alloca %0 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.call @__quantum__rt__qubit_array_allocate(%0, %13, %10) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    %14 = llvm.call @__quantum__rt__qubit_allocate(%10) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb2(%3 : i64)
  ^bb2(%15: i64):  // 2 preds: ^bb1, ^bb3
    %16 = llvm.icmp "slt" %15, %6 : i64
    llvm.cond_br %16, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %17 = llvm.getelementptr %13[%15] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %18 = llvm.load %17 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__h__body(%18) : (!llvm.ptr) -> ()
    %19 = llvm.add %15, %5 : i64
    llvm.br ^bb2(%19 : i64)
  ^bb4:  // pred: ^bb2
    llvm.call @__quantum__qis__x__body(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb5(%3 : i64)
  ^bb5(%20: i64):  // 2 preds: ^bb4, ^bb6
    %21 = llvm.icmp "slt" %20, %6 : i64
    llvm.cond_br %21, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %22 = llvm.getelementptr inbounds|nuw %11[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %23 = llvm.load %22 : !llvm.ptr -> f64
    %24 = llvm.sub %2, %20 : i64
    %25 = llvm.getelementptr %13[%24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %26 = llvm.load %25 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__cp__body(%23, %26, %14) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %27 = llvm.add %20, %5 : i64
    llvm.br ^bb5(%27 : i64)
  ^bb7:  // pred: ^bb5
    llvm.br ^bb8(%3 : i64)
  ^bb8(%28: i64):  // 2 preds: ^bb7, ^bb12
    %29 = llvm.icmp "slt" %28, %6 : i64
    llvm.cond_br %29, ^bb9, ^bb13
  ^bb9:  // pred: ^bb8
    %30 = llvm.sub %28, %5 : i64
    llvm.br ^bb10(%3, %7 : i64, f64)
  ^bb10(%31: i64, %32: f64):  // 2 preds: ^bb9, ^bb11
    %33 = llvm.icmp "slt" %31, %28 : i64
    llvm.cond_br %33, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %34 = llvm.sub %30, %31 : i64
    %35 = llvm.getelementptr %13[%34] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %36 = llvm.load %35 : !llvm.ptr -> !llvm.ptr
    %37 = llvm.getelementptr %13[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %38 = llvm.load %37 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__cp__body(%32, %36, %38) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %39 = llvm.fmul %32, %8 : f64
    %40 = llvm.add %31, %5 : i64
    llvm.br ^bb10(%40, %39 : i64, f64)
  ^bb12:  // pred: ^bb10
    %41 = llvm.getelementptr %13[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %42 = llvm.load %41 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__h__body(%42) : (!llvm.ptr) -> ()
    %43 = llvm.add %28, %5 : i64
    llvm.br ^bb8(%43 : i64)
  ^bb13:  // pred: ^bb8
    llvm.br ^bb14(%3 : i64)
  ^bb14(%44: i64):  // 2 preds: ^bb13, ^bb15
    %45 = llvm.icmp "slt" %44, %6 : i64
    llvm.cond_br %45, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %46 = llvm.getelementptr %13[%44] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %47 = llvm.load %46 : !llvm.ptr -> !llvm.ptr
    %48 = llvm.getelementptr %12[%44] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %49 = llvm.load %48 : !llvm.ptr -> !llvm.ptr
    llvm.call @__quantum__qis__mz__body(%47, %49) : (!llvm.ptr, !llvm.ptr) -> ()
    %50 = llvm.add %44, %5 : i64
    llvm.br ^bb14(%50 : i64)
  ^bb16:  // pred: ^bb14
    llvm.call @__quantum__rt__qubit_release(%14) : (!llvm.ptr) -> ()
    llvm.call @__quantum__rt__qubit_array_release(%0, %13) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb17
  ^bb17:  // pred: ^bb16
    llvm.call @__quantum__rt__result_array_record_output(%6, %12, %9) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_array_release(%0, %12) : (i64, !llvm.ptr) -> ()
    llvm.return %1 : i64
  }
  llvm.func @__quantum__rt__initialize(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__qubit_allocate(!llvm.ptr) -> !llvm.ptr
  llvm.func @__quantum__rt__result_array_allocate(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__h__body(!llvm.ptr)
  llvm.func @__quantum__qis__x__body(!llvm.ptr)
  llvm.func @__quantum__qis__cp__body(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) attributes {passthrough = ["irreversible"]}
  llvm.func @__quantum__rt__qubit_release(!llvm.ptr)
  llvm.func @__quantum__rt__qubit_array_release(i64, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_record_output(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__rt__result_array_release(i64, !llvm.ptr)
  llvm.module_flags [#llvm.mlir.module_flag<error, "qir_major_version", 2 : i32>, #llvm.mlir.module_flag<max, "qir_minor_version", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_qubit_management", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_result_management", 1 : i32>, #llvm.mlir.module_flag<error, "backwards_branching", 1 : i32>, #llvm.mlir.module_flag<error, "arrays", 1 : i32>, #llvm.mlir.module_flag<append, "int_computations", ["i64"]>, #llvm.mlir.module_flag<append, "float_computations", ["double"]>]
}
