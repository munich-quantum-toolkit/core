module {
  func.func @main() attributes {mqt.entry_point} {
    %a = qco.alloc : !qco.qubit
    %b = qco.alloc : !qco.qubit
    %r:2 = qco.inv (%u = %a, %v = %b) {
      %x:2 = qco.swap %u, %v : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %x#1, %x#0 : !qco.qubit, !qco.qubit
    } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
    %qa, %m = qco.measure %r#0 : !qco.qubit
    qco.sink %qa : !qco.qubit
    qco.sink %r#1 : !qco.qubit
    return
  }
}
