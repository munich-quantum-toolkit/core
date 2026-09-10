module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %end = arith.constant 3 : index
    %tensor = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
    %bits = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
    %result = scf.for %i = %c0 to %end step %c1
        iter_args(%t = %tensor) -> tensor<2x!qco.qubit> {
      %rest, %q = qtensor.extract %t[%c0] : tensor<2x!qco.qubit>
      %out = qco.x %q : !qco.qubit -> !qco.qubit
      %updated = qtensor.insert %out into %rest[%c0] : tensor<2x!qco.qubit>
      scf.yield %updated : tensor<2x!qco.qubit>
    }
    %rest, %q = qtensor.extract %result[%c0] : tensor<2x!qco.qubit>
    %out, %bit = qco.measure %q : !qco.qubit
    cbit.store %bit, %bits[%c0] : !cbit.reg<1>
    qco.sink %out : !qco.qubit
    qtensor.dealloc %rest : tensor<2x!qco.qubit>
    return %bits : !cbit.reg<1>
  }
}
