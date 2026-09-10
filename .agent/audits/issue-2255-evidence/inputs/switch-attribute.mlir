module {
  func.func @test(%index: index, %input: !qco.qubit) -> !qco.qubit {
    %result = qco.index_switch %index -> !qco.qubit {audit.marker}
    default args(%arg = %input) {
      qco.yield %arg : !qco.qubit
    }
    return %result : !qco.qubit
  }
}
