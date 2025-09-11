// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: mqtcore-opt %s -p convert-mqtopt-to-qssa --split-input-file | FileCheck %s

// Test full conversion from MQTOpt dialect to QSSA dialect

// Allocate a qubit register - converted to QSSA qu.alloc
// CHECK: %{{.*}} = qu.alloc
%qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

// Extract a qubit from the register - converted to QSSA operations
// CHECK: %{{.*}} = qu.alloc
// CHECK: %{{.*}}, %{{.*}} = 
%qreg1, %qubit = "mqtopt.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

// Apply X gate to the qubit - converted to QSSA
// CHECK: %{{.*}} = qssa.gate<#gate.x> %{{.*}}
%qubit1 = "mqtopt.x"(%qubit) : (!mqtopt.Qubit) -> !mqtopt.Qubit

// Measure the qubit - converted to QSSA
// CHECK: %{{.*}} = qu.alloc
// CHECK: %{{.*}} = qssa.measure %{{.*}}
%qubit2, %bit = "mqtopt.measure"(%qubit1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

// Insert qubit back into register - simplified conversion
// CHECK-NOT: mqtopt.insertQubit
%qreg2 = "mqtopt.insertQubit"(%qreg1, %qubit2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

// Deallocate the register - removed in QSSA
// CHECK-NOT: mqtopt.deallocQubitRegister
"mqtopt.deallocQubitRegister"(%qreg2) : (!mqtopt.QubitRegister) -> ()
