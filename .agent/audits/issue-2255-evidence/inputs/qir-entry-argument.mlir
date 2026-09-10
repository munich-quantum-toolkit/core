module {
  func.func @main(%theta: f64) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.rx(%theta) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
