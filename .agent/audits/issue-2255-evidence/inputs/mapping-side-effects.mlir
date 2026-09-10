module {
  func.func private @first(i64)
  func.func private @second()
  func.func @main(%input: i64) attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %a = arith.addi %input, %input : i64
    %b = arith.addi %a, %input : i64
    func.call @first(%b) : (i64) -> ()
    func.call @second() : () -> ()
    %r:2 = qco.swap %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    qco.sink %r#0 : !qco.qubit
    qco.sink %r#1 : !qco.qubit
    return
  }
}
