module {
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    
    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    
    %q0_3, %m0_0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q1_2, %m1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
}