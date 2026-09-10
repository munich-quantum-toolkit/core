module {
  func.func @permutation() {
    %zero = arith.constant 0 : index
    %three = arith.constant 3 : index
    %one = arith.constant 1 : index
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %a, %b = scf.for %i = %zero to %three step %one
        iter_args(%left = %q0, %right = %q1)
        -> (!qco.qubit, !qco.qubit) {
      scf.yield %right, %left : !qco.qubit, !qco.qubit
    }
    qco.sink %a : !qco.qubit
    qco.sink %b : !qco.qubit
    return
  }
}
