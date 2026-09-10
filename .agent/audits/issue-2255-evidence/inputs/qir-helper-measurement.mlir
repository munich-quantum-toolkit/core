module {
  func.func private @measure(%q: !qc.qubit) -> i1 {
    %bit = qc.measure %q : !qc.qubit -> i1
    return %bit : i1
  }
  func.func @main() -> i1 attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %bit = func.call @measure(%q) : (!qc.qubit) -> i1
    qc.dealloc %q : !qc.qubit
    return %bit : i1
  }
}
