// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --debug \
// RUN:   --catalyst-pipeline="builtin.module(mqtopt-to-catalystquantum)" \
// RUN:   %s | FileCheck %s


// CHECK-LABEL: func @bar()
func.func @bar() {
  // CHECK: %cst = arith.constant 3.000000e-01 : f64
  %cst = arith.constant 3.000000e-01 : f64

  // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
  %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

  // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit
  %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qureg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qureg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

  // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit
  // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[H]] : !quantum.bit
  // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
  // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
  %1 = mqtopt.h() %out_qubit : !mqtopt.Qubit
  %2 = mqtopt.x() %1 : !mqtopt.Qubit
  %3 = mqtopt.y() %2 : !mqtopt.Qubit
  %4 = mqtopt.z() %3 : !mqtopt.Qubit

  // CHECK: %[[CNOT_T:.*]], %[[CNOT_C:.*]] = quantum.custom "CNOT"() %[[Z]] ctrls(%[[Q1]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[CY_T:.*]], %[[CY_C:.*]] = quantum.custom "CY"() %[[CNOT_T]] ctrls(%[[CNOT_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[CZ_T:.*]], %[[CZ_C:.*]] =  quantum.custom "CZ"() %[[CY_T]] ctrls(%[[CY_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[SW0:.*]]:2 = quantum.custom "SWAP"() %[[CZ_C]], %[[CZ_T]] : !quantum.bit, !quantum.bit
  // CHECK: %[[TOF_T:.*]], %[[TOF_C:.*]]:2 = quantum.custom "Toffoli"() %[[SW0]]#0 ctrls(%[[Q2]], %[[SW0]]#1) ctrlvals(%true{{.*}}, %true{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %5, %6 = mqtopt.x() %4 ctrl %out_qubit_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %7, %8 = mqtopt.y() %5 ctrl %6 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %9, %10 = mqtopt.z() %7 ctrl %8 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %11, %12 = mqtopt.swap() %10, %9 : !mqtopt.Qubit, !mqtopt.Qubit
    %13, %14, %15 = mqtopt.x() %11 ctrl %out_qubit_3, %12 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RX:.*]] = quantum.custom "RX"(%cst) %[[TOF_T]] : !quantum.bit
  // CHECK: %[[RY:.*]] = quantum.custom "RY"(%cst) %[[RX]] : !quantum.bit
  // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%cst) %[[RY]] : !quantum.bit
  // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"(%cst) %[[RZ]] : !quantum.bit
  %16 = mqtopt.rx(%cst) %13 : !mqtopt.Qubit
  %17 = mqtopt.ry(%cst) %16 : !mqtopt.Qubit
  %18 = mqtopt.rz(%cst) %17 : !mqtopt.Qubit
  %19 = mqtopt.p(%cst) %18 : !mqtopt.Qubit

  // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = quantum.custom "CRX"(%cst) %[[PS]] ctrls(%[[TOF_C]]#0) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = quantum.custom "CRY"(%cst) %[[CRX_T]] ctrls(%[[CRX_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[CRZ_T:.*]], %[[CRZ_C:.*]] = quantum.custom "CRZ"(%cst) %[[CRY_T]] ctrls(%[[CRY_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[CPS_T:.*]], %[[CPS_C:.*]] = quantum.custom "ControlledPhaseShift"(%cst) %[[CRZ_T]] ctrls(%[[CRZ_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
  %200, %201 = mqtopt.rx(%cst) %19 ctrl %14 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %210, %211 = mqtopt.ry(%cst) %200 ctrl %201 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %220, %221 = mqtopt.rz(%cst) %210 ctrl %211 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %230, %231 = mqtopt.p(%cst) %220 ctrl %221 : !mqtopt.Qubit ctrl !mqtopt.Qubit

  // CHECK: %[[XY:.*]]:2 = quantum.custom "IsingXY"(%cst, %cst) %[[CPS_T]], %[[CPS_C]] : !quantum.bit, !quantum.bit
  // CHECK: %[[H1:.*]] = quantum.custom "Hadamard"() %[[XY]]#1 : !quantum.bit
  // CHECK: %[[RZZ:.*]]:2 = quantum.custom "IsingZZ"(%cst) %[[XY]]#0, %[[H1]] : !quantum.bit, !quantum.bit
  // CHECK: %[[H2:.*]] = quantum.custom "Hadamard"() %[[RZZ]]#1 : !quantum.bit
  %xy:2 = mqtopt.xxplusyy(%cst, %cst) %230, %231 : !mqtopt.Qubit, !mqtopt.Qubit
  %rzx0, %rzx1 = mqtopt.rzx(%cst) %xy#0, %xy#1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[C1:.*]]:2 = quantum.custom "CNOT"() {{%.*}}, {{%.*}} : !quantum.bit, !quantum.bit
  // CHECK: %[[C2:.*]]:2 = quantum.custom "CNOT"() %[[C1]]#1, %[[C1]]#0 : !quantum.bit, !quantum.bit
  %dcx0, %dcx1 = mqtopt.dcx() %rzx0, %rzx1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[PI2:.*]] = arith.constant 1.5707963267948966 : f64
  // First, check the explicit Z–Y–Z sequence on some input qubit.
  // CHECK: %[[RZ1_OUT:.*]] = quantum.custom "RZ"() %[[C2]]#0 : !quantum.bit
  // CHECK: %[[RY_OUT:.*]]  = quantum.custom "RY"() %[[RZ1_OUT]] : !quantum.bit
  // CHECK: %[[RZ2_OUT:.*]] = quantum.custom "RZ"() %[[RY_OUT]] adj : !quantum.bit
  %v = mqtopt.v() %dcx0 : !mqtopt.Qubit

  // CHECK: %[[NPI2:.*]] = arith.constant -1.5707963267948966 : f64
  // CHECK: %[[VDG_RZ1:.*]] = quantum.custom "RZ"(%[[NPI2]]) %[[RZ2_OUT]] adj : !quantum.bit
  // CHECK: %[[VDG_RY:.*]] = quantum.custom "RY"(%[[NPI2]]) %[[VDG_RZ1]] : !quantum.bit
  // CHECK: %[[VDG_RZ2:.*]] = quantum.custom "RZ"(%[[NPI2]]) %[[VDG_RY]] : !quantum.bit
  %vdg = mqtopt.vdg() %v : !mqtopt.Qubit

  // CHECK: %[[S_OUT:.*]] = quantum.custom "S"() %[[VDG_RZ2]] : !quantum.bit
  %s = mqtopt.s() %vdg : !mqtopt.Qubit

  // CHECK: %[[SDG_OUT:.*]] = quantum.custom "S"() %[[S_OUT]] adj : !quantum.bit
  %sdg = mqtopt.sdg() %s : !mqtopt.Qubit

  // CHECK: %[[T_OUT:.*]] = quantum.custom "T"() %[[SDG_OUT]] : !quantum.bit
  %t = mqtopt.t() %sdg : !mqtopt.Qubit

  // CHECK: %[[TDG_OUT:.*]] = quantum.custom "T"() %[[T_OUT]] adj : !quantum.bit
  %tdg = mqtopt.tdg() %t : !mqtopt.Qubit

  // CHECK: %[[ISWAP_OUT:.*]]:2 = quantum.custom "ISWAP"() %[[TDG_OUT]], %[[C2]]#1 : !quantum.bit, !quantum.bit
  %iswap0, %iswap1 = mqtopt.iswap() %tdg, %dcx1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[ISWAPDAG_OUT:.*]]:2 = quantum.custom "ISWAP"() %[[ISWAP_OUT]]#0, %[[ISWAP_OUT]]#1 adj : !quantum.bit, !quantum.bit
  %iswapdag0, %iswapdag1 = mqtopt.iswapdg() %iswap0, %iswap1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[ECR_OUT:.*]]:2 = quantum.custom "ECR"() %[[ISWAPDAG_OUT]]#0, %[[ISWAPDAG_OUT]]#1 : !quantum.bit, !quantum.bit
  %ecr0, %ecr1 = mqtopt.ecr() %iswapdag0, %iswapdag1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RXX_OUT:.*]]:2 = quantum.custom "IsingXX"(%cst) %[[ECR_OUT]]#0, %[[ECR_OUT]]#1 : !quantum.bit, !quantum.bit
  %rxx0, %rxx1 = mqtopt.rxx(%cst) %ecr0, %ecr1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RYY_OUT:.*]]:2 = quantum.custom "IsingYY"(%cst) %[[RXX_OUT]]#0, %[[RXX_OUT]]#1 : !quantum.bit, !quantum.bit
  %ryy0, %ryy1 = mqtopt.ryy(%cst) %rxx0, %rxx1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RZZ_OUT:.*]]:2 = quantum.custom "IsingZZ"(%cst) %[[RYY_OUT]]#0, %[[RYY_OUT]]#1 : !quantum.bit, !quantum.bit
  %rzz0, %rzz1 = mqtopt.rzz(%cst) %ryy0, %ryy1 : !mqtopt.Qubit, !mqtopt.Qubit

  // XXminusYY is decomposed into a series of rotations and CNOTs
  // CHECK: %[[PI2_XXMY:.*]] = arith.constant 1.5707963267948966 : f64
  // CHECK: %[[NPI2_XXMY:.*]] = arith.constant -1.5707963267948966 : f64
  // CHECK: %[[RX1_XXMY:.*]]:2 = quantum.custom "RX"(%[[PI2_XXMY]]) %[[RZZ_OUT]]#0, %[[RZZ_OUT]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[RY1_XXMY:.*]]:2 = quantum.custom "RY"(%[[PI2_XXMY]]) %[[RX1_XXMY]]#0, %[[RX1_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[CNOT1_XXMY:.*]]:2 = quantum.custom "CNOT"() %[[RY1_XXMY]]#0, %[[RY1_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[RZ1_XXMY:.*]]:2 = quantum.custom "RZ"(%cst) %[[CNOT1_XXMY]]#0, %[[CNOT1_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[CNOT2_XXMY:.*]]:2 = quantum.custom "CNOT"() %[[RZ1_XXMY]]#0, %[[RZ1_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[RZ2_XXMY:.*]]:2 = quantum.custom "RZ"(%cst) %[[CNOT2_XXMY]]#0, %[[CNOT2_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[RZ3_XXMY:.*]]:2 = quantum.custom "RZ"(%cst) %[[RZ2_XXMY]]#0, %[[RZ2_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[RX2_XXMY:.*]]:2 = quantum.custom "RX"(%[[NPI2_XXMY]]) %[[RZ3_XXMY]]#0, %[[RZ3_XXMY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[RY2_XXMY:.*]]:2 = quantum.custom "RY"(%[[NPI2_XXMY]]) %[[RX2_XXMY]]#0, %[[RX2_XXMY]]#1 : !quantum.bit, !quantum.bit
  %xxmy0, %xxmy1 = mqtopt.xxminusyy(%cst, %cst) %rzz0, %rzz1 : !mqtopt.Qubit, !mqtopt.Qubit

  // GlobalPhase doesn't take or return qubits and uses quantum.gphase directly
  // CHECK: quantum.gphase(%cst) :
  mqtopt.gphase(%cst) : ()

  // CHECK: %[[I_OUT:.*]] = quantum.custom "Identity"() %[[RY2_XXMY]]#0 : !quantum.bit
  %i = mqtopt.i() %xxmy0 : !mqtopt.Qubit

  // CHECK: %[[U_RZ1:.*]] = quantum.custom "RZ"({{.*}}) %[[I_OUT]] : !quantum.bit
  // CHECK: %[[U_RX1:.*]] = quantum.custom "RX"({{.*}}) %[[U_RZ1]] : !quantum.bit
  // CHECK: %[[U_RZ2:.*]] = quantum.custom "RZ"({{.*}}) %[[U_RX1]] : !quantum.bit
  // CHECK: %[[U_RX2:.*]] = quantum.custom "RX"({{.*}}) %[[U_RZ2]] : !quantum.bit
  // CHECK: %[[U_OUT:.*]] = quantum.custom "RZ"({{.*}}) %[[U_RX2]] : !quantum.bit
  %u = mqtopt.u(%cst, %cst, %cst) %i : !mqtopt.Qubit

  // CHECK: %[[U2_RZ1:.*]] = quantum.custom "RZ"({{.*}}) %[[U_OUT]] : !quantum.bit
  // CHECK: %[[U2_RX1:.*]] = quantum.custom "RX"({{.*}}) %[[U2_RZ1]] : !quantum.bit
  // CHECK: %[[U2_RZ2:.*]] = quantum.custom "RZ"({{.*}}) %[[U2_RX1]] : !quantum.bit
  // CHECK: %[[U2_RX2:.*]] = quantum.custom "RX"({{.*}}) %[[U2_RZ2]] : !quantum.bit
  // CHECK: %[[U2_OUT:.*]] = quantum.custom "RZ"({{.*}}) %[[U2_RX2]] : !quantum.bit
  %u2 = mqtopt.u2(%cst, %cst) %u : !mqtopt.Qubit

  // CHECK: %[[MRES:.*]], %[[QMEAS:.*]] = quantum.measure %[[U2_OUT]] : i1, !quantum.bit
  %q_meas, %c0_0 = "mqtopt.measure"(%u2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

  // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 2], %[[QMEAS]] : !quantum.reg, !quantum.bit
  // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[RZZ]]#0 : !quantum.reg, !quantum.bit
  // CHECK: %[[R3:.*]] = quantum.insert %[[R2]][ 0], %[[H2]] : !quantum.reg, !quantum.bit
  %240 = "mqtopt.insertQubit"(%out_qureg_2, %q_meas) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %250 = "mqtopt.insertQubit"(%240, %rzx0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %260 = "mqtopt.insertQubit"(%250, %rzx1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

  // CHECK: quantum.dealloc %[[R3]] : !quantum.reg
  "mqtopt.deallocQubitRegister"(%260) : (!mqtopt.QubitRegister) -> ()

  return
}
