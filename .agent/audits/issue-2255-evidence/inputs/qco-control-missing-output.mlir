module {
  func.func @test(%control: !qco.qubit, %target: !qco.qubit) -> !qco.qubit {
    %result = "qco.ctrl"(%control, %target) <{
      operandSegmentSizes = array<i32: 1, 1>,
      resultSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%arg: !qco.qubit):
      %acted = qco.x %arg : !qco.qubit -> !qco.qubit
      qco.yield %acted : !qco.qubit
    }) : (!qco.qubit, !qco.qubit) -> !qco.qubit
    return %result : !qco.qubit
  }
}
