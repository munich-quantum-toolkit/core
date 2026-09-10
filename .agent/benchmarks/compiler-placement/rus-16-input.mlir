module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = qc.alloc : !qc.qubit
    %alloc = memref.alloc() {mqt.register_name = "data"} : memref<16x!qc.qubit>
    %1 = memref.load %alloc[%c0] : memref<16x!qc.qubit>
    %2 = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "result"} : !cbit.reg<1>
    scf.while : () -> () {
      qc.h %0 : !qc.qubit
      qc.t %0 : !qc.qubit
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %5 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
        qc.ctrl(%0) targets (%arg1 = %5) {
          qc.x %arg1 : !qc.qubit
          qc.yield
        } : {!qc.qubit}, {!qc.qubit}
      }
      qc.h %0 : !qc.qubit
      scf.for %arg0 = %c0 to %c16 step %c1 {
        %5 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
        qc.ctrl(%0) targets (%arg1 = %5) {
          qc.x %arg1 : !qc.qubit
          qc.yield
        } : {!qc.qubit}, {!qc.qubit}
      }
      qc.t %0 : !qc.qubit
      qc.h %0 : !qc.qubit
      %4 = qc.measure %0 : !qc.qubit -> i1
      scf.condition(%4)
    } do {
      qc.x %0 : !qc.qubit
      scf.yield
    }
    qc.sdg %1 : !qc.qubit
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %4 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
      qc.h %4 : !qc.qubit
    }
    scf.for %arg0 = %c1 to %c16 step %c1 {
      %4 = memref.load %alloc[%arg0] : memref<16x!qc.qubit>
      qc.ctrl(%4) targets (%arg1 = %1) {
        qc.x %arg1 : !qc.qubit
        qc.yield
      } : {!qc.qubit}, {!qc.qubit}
    }
    %3 = qc.measure %1 : !qc.qubit -> i1
    cbit.store %3, %2[%c0] : !cbit.reg<1>
    qc.dealloc %0 : !qc.qubit
    memref.dealloc %alloc : memref<16x!qc.qubit>
    return %2 : !cbit.reg<1>
  }
}
