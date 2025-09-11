// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: MQT_ROUNDTRIP

// Test qubit allocation, extraction, X gate, and deallocation
// Allocate a single qubit register with size 1
// CHECK: %qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
// CHECK-GENERIC: %qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
%qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

// Extract a qubit from the register
// CHECK: %qreg1, %qubit = "mqtopt.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
// CHECK-GENERIC: %qreg1, %qubit = "mqtopt.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
%qreg1, %qubit = "mqtopt.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

// Apply X gate to the extracted qubit (value semantics)
// CHECK: %qubit1 = "mqtopt.x"(%qubit) : (!mqtopt.Qubit) -> !mqtopt.Qubit
// CHECK-GENERIC: %qubit1 = "mqtopt.x"(%qubit) : (!mqtopt.Qubit) -> !mqtopt.Qubit
%qubit1 = "mqtopt.x"(%qubit) : (!mqtopt.Qubit) -> !mqtopt.Qubit

// Measure the qubit
// CHECK: %qubit2, %bit = "mqtopt.measure"(%qubit1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
// CHECK-GENERIC: %qubit2, %bit = "mqtopt.measure"(%qubit1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
%qubit2, %bit = "mqtopt.measure"(%qubit1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

// Insert the qubit back into the register (for completeness, though not strictly needed for dealloc)
// CHECK: %qreg2 = "mqtopt.insertQubit"(%qreg1, %qubit2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
// CHECK-GENERIC: %qreg2 = "mqtopt.insertQubit"(%qreg1, %qubit2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
%qreg2 = "mqtopt.insertQubit"(%qreg1, %qubit2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

// Deallocate the register
// CHECK: "mqtopt.deallocQubitRegister"(%qreg2) : (!mqtopt.QubitRegister) -> ()
// CHECK-GENERIC: "mqtopt.deallocQubitRegister"(%qreg2) : (!mqtopt.QubitRegister) -> ()
"mqtopt.deallocQubitRegister"(%qreg2) : (!mqtopt.QubitRegister) -> ()
