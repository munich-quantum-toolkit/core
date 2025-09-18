// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef @DIALECT_NAME_UPPER@_STD_OPS
#define @DIALECT_NAME_UPPER@_STD_OPS

include "mlir/Dialect/@DIALECT_NAME@/IR/@DIALECT_NAME@Ops.td"

def GPhaseOp : UnitaryOp<"gphase", [NoTarget, OneParameter]> {
    let summary = "GPhase operation";

    let description = [{
        This class represents a global phase gate. It accepts a parameter indicating
        the degree of the rotation angle. The global phase gate does not have
        any input and output qubits. It might still be controlled by arbitrarily
        many qubits.
    }];
}

def IOp : UnitaryOp<"i", [OneTarget, NoParameter]> {
    let summary = "I operation";

    let description = [{
        This class represents an identity gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def BarrierOp : UnitaryOp<"barrier", [NoParameter, NoControl]> {
    let summary = "Barrier operation";

    let description = [{
        This class represents the barrier operation. It has a variadic number
        of input qubits and no control qubits.
    }];
}

def HOp : UnitaryOp<"h", [OneTarget, NoParameter]> {
    let summary = "H operation";

    let description = [{
        This class represents a Hadamard gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def XOp : UnitaryOp<"x", [OneTarget, NoParameter]> {
    let summary = "X operation";

    let description = [{
        This class represents a Pauli-X gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def YOp : UnitaryOp<"y", [OneTarget, NoParameter]> {
    let summary = "Y operation";

    let description = [{
        This class represents a Pauli-Y gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def ZOp : UnitaryOp<"z", [OneTarget, NoParameter]> {
    let summary = "Z operation";

    let description = [{
        This class represents a Pauli-Z gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def SOp : UnitaryOp<"s", [OneTarget, NoParameter]> {
    let summary = "S operation";

    let description = [{
        This class represents an S gate. It takes a qubit as an input and a
        variadic list of positive/negative controls.
    }];
}

def SdgOp : UnitaryOp<"sdg", [OneTarget, NoParameter]> {
    let summary = "Sdg operation";

    let description = [{
        This class represents an inverse S gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def TOp : UnitaryOp<"t", [OneTarget, NoParameter]> {
    let summary = "T operation";

    let description = [{
        This class represents a T gate. It takes a qubit and a variadic list of
        positive/negative controls as an input.
    }];
}

def TdgOp : UnitaryOp<"tdg", [OneTarget, NoParameter]> {
    let summary = "Tdg operation";

    let description = [{
        This class represents an inverse T gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def VOp : UnitaryOp<"v", [OneTarget, NoParameter]> {
    let summary = "V operation";

    let description = [{
        This class represents a V gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def VdgOp : UnitaryOp<"vdg", [OneTarget, NoParameter]> {
    let summary = "Vdg operation";

    let description = [{
        This class represents an inverse V gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def UOp : UnitaryOp<"u", [OneTarget, ThreeParameters]> {
    let summary = "U operation";

    let description = [{
        This class represents a U gate. It takes a qubit and a variadic
        list of positive/negative controls as an input. Additionally, it accepts
        three parameters indicating the degree of the rotation angles.
    }];
}

def U2Op : UnitaryOp<"u2", [OneTarget, TwoParameters]> {
    let summary = "U2 operation";

    let description = [{
        This class represents a U2 gate. It takes a qubit and a variadic
        list of positive/negative controls as an input. Additionally, it accepts
        two parameters indicating the degree of the rotation angles.
    }];
}

def POp : UnitaryOp<"p", [OneTarget, OneParameter]> {
    let summary = "P operation";

    let description = [{
        This class represents a phase gate. It takes a qubit and a variadic
        list of positive/negative controls as an input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.
    }];
}

def SXOp : UnitaryOp<"sx", [OneTarget, NoParameter]> {
    let summary = "SX operation";

    let description = [{
        This class represents an SX gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.
    }];
}

def SXdgOp : UnitaryOp<"sxdg", [OneTarget, NoParameter]> {
    let summary = "SXdg operation";

    let description = [{
        This class represents an inverse SX gate. It takes a qubit and a
        variadic list of positive/negative controls as an input.
    }];
}

def RXOp : UnitaryOp<"rx", [OneTarget, OneParameter]> {
    let summary = "RX operation";

    let description = [{
        This class represents an RX gate. It takes a qubit and a variadic
        list of positive/negative controls as input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.
    }];
}

def RYOp : UnitaryOp<"ry", [OneTarget, OneParameter]> {
    let summary = "RY operation";

    let description = [{
        This class represents an RY gate. It takes a qubit and a variadic
        list of positive/negative controls as input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.
    }];
}

def RZOp : UnitaryOp<"rz", [OneTarget, OneParameter]> {
    let summary = "RZ operation";

    let description = [{
        This class represents an RZ gate. It takes a qubit and a variadic
        list of positive/negative controls as input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.
    }];
}

def SWAPOp : UnitaryOp<"swap", [TwoTarget, NoParameter]> {
    let summary = "SWAP operation";

    let description = [{
        This class represents a SWAP gate. It takes two qubits and a variadic
        list of positive/negative controls as input.
    }];
}

def iSWAPOp : UnitaryOp<"iswap", [TwoTarget, NoParameter]> {
    let summary = "iSWAP operation";

    let description = [{
        This class represents an iSWAP gate. It takes two qubits and a variadic
        list of positive/negative controls as input.
    }];
}

def iSWAPdgOp : UnitaryOp<"iswapdg", [TwoTarget, NoParameter]> {
    let summary = "iSWAPdg operation";

    let description = [{
        This class represents an inverse iSWAP gate. It takes two qubits and a
        variadic list of positive/negative controls as input.
    }];
}

def PeresOp : UnitaryOp<"peres", [TwoTarget, NoParameter]> {
    let summary = "Peres operation";

    let description = [{
        This class represents a Peres gate. It takes two qubits and a variadic
        list of positive/negative controls as input.
    }];
}

def PeresdgOp : UnitaryOp<"peresdg", [TwoTarget, NoParameter]> {
    let summary = "Peresdg operation";

    let description = [{
        This class represents an inverse Peres gate. It takes two qubits and a
        variadic list of positive/negative controls as input.
    }];
}

def DCXOp : UnitaryOp<"dcx", [TwoTarget, NoParameter]> {
    let summary = "DCX operation";

    let description = [{
        This class represents a DCX gate. It takes two qubits and a variadic
        list of positive/negative controls as input.
    }];
}

def ECROp : UnitaryOp<"ecr", [TwoTarget, NoParameter]> {
    let summary = "DCX operation";

    let description = [{
        This class represents an ECR gate. It takes two qubits and a variadic
        list of positive/negative controls as input.
    }];
}

def RXXOp : UnitaryOp<"rxx", [TwoTarget, OneParameter]> {
    let summary = "RXX operation";

    let description = [{
        This class represents an RXX gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.
    }];
}

def RYYOp : UnitaryOp<"ryy", [TwoTarget, OneParameter]> {
    let summary = "RYY operation";

    let description = [{
        This class represents an RYY gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.
    }];
}

def RZZOp : UnitaryOp<"rzz", [TwoTarget, OneParameter]> {
    let summary = "RZZ operation";

    let description = [{
        This class represents an RZZ gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.
    }];
}

def RZXOp : UnitaryOp<"rzx", [TwoTarget, OneParameter]> {
    let summary = "RZX operation";

    let description = [{
        This class represents an RZX gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.
    }];
}

def XXminusYYOp : UnitaryOp<"xx_minus_yy", [TwoTarget, TwoParameters]> {
    let summary = "XX-YY operation";

    let description = [{
        This class represents an XX-YY gate. It takes two qubits and a
        variadic list of positive/negative controls as input. Additionally, it
        accepts two parameters indicating the degree of the rotation angles.
    }];
}

def XXplusYYOp : UnitaryOp<"xx_plus_yy", [TwoTarget, TwoParameters]> {
    let summary = "XX+YY operation";

    let description = [{
        This class represents an XX+YY gate. It takes two qubits and a
        variadic list of positive/negative controls as input. Additionally, it
        accepts two parameters indicating the degree of the rotation angles.
    }];
}

#endif // @DIALECT_NAME_UPPER@_STD_OPS
