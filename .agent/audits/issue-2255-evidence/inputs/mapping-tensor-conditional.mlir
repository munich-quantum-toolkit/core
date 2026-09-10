module {
  func.func @main(%condition: i1) attributes {mqt.entry_point} {
    %size = arith.constant 1 : index
    %reg = qtensor.alloc(%size) : tensor<1x!qco.qubit>
    %out = qco.if %condition args(%arg = %reg) -> (tensor<1x!qco.qubit>) {
      qco.yield %arg : tensor<1x!qco.qubit>
    } else args(%arg = %reg) {
      qco.yield %arg : tensor<1x!qco.qubit>
    }
    qtensor.dealloc %out : tensor<1x!qco.qubit>
    return
  }
}
