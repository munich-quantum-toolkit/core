// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: mqtcore-opt %s -p convert-mqtopt-to-qssa --split-input-file | FileCheck %s

// CHECK-LABEL: builtin.module {
// CHECK: %qubit = qu.alloc
// CHECK: %qubit1 = qssa.gate<#gate.x> %qubit
// CHECK: %bit = qssa.measure %qubit1
// CHECK: %qubit2 = qu.alloc
// CHECK: }

// Test full conversion from MQTOpt dialect to QSSA dialect

%qreg = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

%qreg1, %qubit = "mqtopt.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

%qubit1 = "mqtopt.x"(%qubit) : (!mqtopt.Qubit) -> !mqtopt.Qubit

%qubit2, %bit = "mqtopt.measure"(%qubit1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

%qreg2 = "mqtopt.insertQubit"(%qreg1, %qubit2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

"mqtopt.deallocQubitRegister"(%qreg2) : (!mqtopt.QubitRegister) -> ()
