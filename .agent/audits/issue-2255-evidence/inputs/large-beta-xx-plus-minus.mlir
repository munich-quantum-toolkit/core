module {
  func.func @plus(%q0: !qco.qubit, %q1: !qco.qubit)
      -> (!qco.qubit, !qco.qubit) {
    %theta = arith.constant 1.0 : f64
    %beta = arith.constant 1.0e16 : f64
    %r0, %r1 = qco.xx_plus_yy(%theta, %beta) %q0, %q1
        : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    return %r0, %r1 : !qco.qubit, !qco.qubit
  }
  func.func @minus(%q0: !qco.qubit, %q1: !qco.qubit)
      -> (!qco.qubit, !qco.qubit) {
    %theta = arith.constant 1.0 : f64
    %beta = arith.constant 1.0e16 : f64
    %r0, %r1 = qco.xx_minus_yy(%theta, %beta) %q0, %q1
        : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    return %r0, %r1 : !qco.qubit, !qco.qubit
  }
}
