module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %c1 = arith.constant 1 : index
    %0 = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
    %c0 = arith.constant 0 : index
    %out_tensor, %result = qtensor.extract %0[%c0] : tensor<1x!qco.qubit>
    %true = arith.constant true
    %linearResults:2 = qco.if %true args(%arg0 = %out_tensor, %arg1 = %result) -> (tensor<1x!qco.qubit>, !qco.qubit) {
      %c0_0 = arith.constant 0 : index
      %2 = qtensor.insert %arg1 into %arg0[%c0_0] : tensor<1x!qco.qubit>
      %c0_1 = arith.constant 0 : index
      %out_tensor_2, %result_3 = qtensor.extract %2[%c0_1] : tensor<1x!qco.qubit>
      qco.yield %out_tensor_2, %result_3 : tensor<1x!qco.qubit>, !qco.qubit
    } else args(%arg0 = %out_tensor, %arg1 = %result) {
      qco.yield %arg0, %arg1 : tensor<1x!qco.qubit>, !qco.qubit
    }
    %c0_i64 = arith.constant 0 : i64
    %1 = qtensor.insert %linearResults#1 into %linearResults#0[%c0] : tensor<1x!qco.qubit>
    qtensor.dealloc %1 : tensor<1x!qco.qubit>
    return %c0_i64 : i64
  }
}
