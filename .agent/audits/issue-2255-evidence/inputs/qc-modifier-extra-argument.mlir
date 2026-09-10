module {
  func.func @test(%input: !qc.qubit) {
    "qc.inv"(%input) ({
    ^bb0(%first: !qc.qubit, %extra: !qc.qubit):
      qc.x %extra : !qc.qubit
      qc.yield
    }) : (!qc.qubit) -> ()
    return
  }
}
