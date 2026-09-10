module {
  func.func @measure(%q: !qco.qubit) -> (!qco.qubit, i1) {
    %out, %bit = qco.measure %q : !qco.qubit
    return %out, %bit : !qco.qubit, i1
  }
}
