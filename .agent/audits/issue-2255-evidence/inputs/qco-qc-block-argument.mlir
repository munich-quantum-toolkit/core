module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.alloc : !qco.qubit
    cf.br ^next(%q : !qco.qubit)
  ^next(%arg: !qco.qubit):
    %out = qco.h %arg : !qco.qubit -> !qco.qubit
    qco.sink %out : !qco.qubit
    return
  }
}
