// Base Profile Programs.
module {
    // The bell state.
    // func.func @bell() {
    //     %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    //     %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    //     %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        
        
    //     %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    //     %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        
    //     %q0_3, %m0_0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    //     %q1_2, %m1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        
    //     %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    //     %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister     
    //     "mqtopt.deallocQubitRegister"(%reg_0) : (!mqtopt.QubitRegister) -> ()
    //     return
    // }

    // Figure 4 in SABRE Paper "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices".
    func.func @sabre() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 6 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_4, %q3_0 = "mqtopt.extractQubit"(%reg_3) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_5, %q4_0 = "mqtopt.extractQubit"(%reg_4) <{index_attr = 4 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_6, %q5_0 = "mqtopt.extractQubit"(%reg_5) <{index_attr = 5 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        

        %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
        %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit
        %q4_1 = mqtopt.h() %q4_0 : !mqtopt.Qubit

        %q0_2 = mqtopt.z() %q0_1 : !mqtopt.Qubit
        %q2_1, %q1_2 = mqtopt.x() %q2_0 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g1
        %q5_1, %q4_2 = mqtopt.x() %q5_0 ctrl %q4_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g2

        %q1_3, %q0_3 = mqtopt.x() %q1_2 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g3
        %q3_1, %q2_2 = mqtopt.x() %q3_0 ctrl %q2_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g4

        %q2_3 = mqtopt.h() %q2_2 : !mqtopt.Qubit
        %q3_2 = mqtopt.h() %q3_1 : !mqtopt.Qubit

        %q2_4, %q1_4 = mqtopt.x() %q2_3 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g5
        %q5_2, %q3_3 = mqtopt.x() %q5_1 ctrl %q3_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g6

        %q3_4 = mqtopt.z() %q3_3 : !mqtopt.Qubit

        %q3_5, %q4_3 = mqtopt.x() %q3_4 ctrl %q4_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g7

        %q0_4, %q3_6 = mqtopt.x() %q0_3 ctrl %q3_5 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g8


        %reg_7 = "mqtopt.insertQubit"(%reg_6, %q0_4) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_8 = "mqtopt.insertQubit"(%reg_7, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_9 = "mqtopt.insertQubit"(%reg_8, %q2_4) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_10 = "mqtopt.insertQubit"(%reg_9, %q3_6) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_11 = "mqtopt.insertQubit"(%reg_10, %q4_3) <{index_attr = 4 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_12 = "mqtopt.insertQubit"(%reg_11, %q5_2) <{index_attr = 5 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_12) : (!mqtopt.QubitRegister) -> ()
        
        return
    }
}