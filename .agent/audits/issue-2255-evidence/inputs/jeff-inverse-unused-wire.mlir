module {
  func.func @main() attributes {mqt.entry_point} {
    %a = qco.alloc : !qco.qubit
    %b = qco.alloc : !qco.qubit
    %r:2 = qco.inv (%u = %a, %v = %b) {
      %x = qco.s %u : !qco.qubit -> !qco.qubit
      qco.yield %x, %v : !qco.qubit, !qco.qubit
    } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
    qco.sink %r#0 : !qco.qubit
    qco.sink %r#1 : !qco.qubit
    return
  }
}
