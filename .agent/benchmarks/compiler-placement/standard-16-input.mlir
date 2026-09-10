module {
  func.func @main() -> !cbit.reg<16> attributes {mqt.entry_point} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant -1.5707963267948966 : f64
    %cst_1 = arith.constant dense<[2.3561944901923448, 4.7123889803846897, 3.1415926535897931, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<16xf64>
    %c15 = arith.constant 15 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {mqt.register_name = "query"} : memref<16x!qc.qubit>
    %0 = qc.alloc : !qc.qubit
    %1 = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "result"} : !cbit.reg<16>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %2 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
      qc.h %2 : !qc.qubit
    }
    qc.x %0 : !qc.qubit
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %extracted = tensor.extract %cst_1[%arg0] : tensor<16xf64>
      %2 = arith.subi %c15, %arg0 : index
      %3 = memref.load %alloc[%2] : memref<16x!qc.qubit>
      qc.ctrl(%3) targets (%arg1 = %0) {
        qc.p(%extracted) %arg1 : !qc.qubit
        qc.yield
      } : {!qc.qubit}, {!qc.qubit}
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %2 = arith.subi %arg0, %c1 : index
      %3 = scf.for %arg1 = %c0 to %arg0 step %c1 iter_args(%arg2 = %cst_0) -> (f64) {
        %5 = arith.subi %2, %arg1 : index
        %6 = memref.load %alloc[%5] : memref<16x!qc.qubit>
        %7 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
        qc.ctrl(%6) targets (%arg3 = %7) {
          qc.p(%arg2) %arg3 : !qc.qubit
          qc.yield
        } : {!qc.qubit}, {!qc.qubit}
        %8 = arith.mulf %arg2, %cst : f64
        scf.yield %8 : f64
      }
      %4 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
      qc.h %4 : !qc.qubit
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %2 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
      %3 = qc.measure %2 : !qc.qubit -> i1
      cbit.store %3, %1[%arg0] : !cbit.reg<16>
    }
    qc.dealloc %0 : !qc.qubit
    memref.dealloc %alloc : memref<16x!qc.qubit>
    return %1 : !cbit.reg<16>
  }
}
