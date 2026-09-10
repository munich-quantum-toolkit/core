"builtin.module"() ({
  "func.func"() <{function_type = () -> i64, sym_name = "main"}> ({
    %0 = "arith.constant"() <{value = 1 : index}> : () -> index
    %1 = "qtensor.alloc"(%0) : (index) -> tensor<1x!qco.qubit>
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3:2 = "qtensor.extract"(%1, %2) : (tensor<1x!qco.qubit>, index) -> (tensor<1x!qco.qubit>, !qco.qubit)
    %4 = "arith.constant"() <{value = true}> : () -> i1
    %5:2 = "qco.if"(%4, %3#0, %3#1) <{resultSegmentSizes = array<i32: 0, 2>}> ({
    ^bb0(%arg2: tensor<1x!qco.qubit>, %arg3: !qco.qubit):
      %8 = "arith.constant"() <{value = 0 : index}> : () -> index
      %9 = "qtensor.insert"(%arg3, %arg2, %8) : (!qco.qubit, tensor<1x!qco.qubit>, index) -> tensor<1x!qco.qubit>
      %10 = "arith.constant"() <{value = 0 : index}> : () -> index
      %11:2 = "qtensor.extract"(%9, %10) : (tensor<1x!qco.qubit>, index) -> (tensor<1x!qco.qubit>, !qco.qubit)
      "qco.yield"(%11#0, %11#1) : (tensor<1x!qco.qubit>, !qco.qubit) -> ()
    }, {
    ^bb0(%arg0: tensor<1x!qco.qubit>, %arg1: !qco.qubit):
      "qco.yield"(%arg0, %arg1) : (tensor<1x!qco.qubit>, !qco.qubit) -> ()
    }) : (i1, tensor<1x!qco.qubit>, !qco.qubit) -> (tensor<1x!qco.qubit>, !qco.qubit)
    %6 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %7 = "qtensor.insert"(%5#1, %5#0, %10) : (!qco.qubit, tensor<1x!qco.qubit>, index) -> tensor<1x!qco.qubit>
    "qtensor.dealloc"(%7) : (tensor<1x!qco.qubit>) -> ()
    "func.return"(%6) : (i64) -> ()
  }) {mqt.entry_point} : () -> ()
}) : () -> ()
