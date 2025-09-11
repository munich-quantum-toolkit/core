// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: MQT_ROUNDTRIP

// Test simple qubit allocation, X gate, and deallocation with reference semantics
// Allocate a single qubit
// CHECK: %qubit = "mqtref.allocQubit"() : () -> !mqtref.Qubit
// CHECK-GENERIC: %qubit = "mqtref.allocQubit"() : () -> !mqtref.Qubit
%qubit = "mqtref.allocQubit"() : () -> !mqtref.Qubit

// Apply X gate (reference semantics - operates in place)
// CHECK: "mqtref.x"(%qubit) : (!mqtref.Qubit) -> ()
// CHECK-GENERIC: "mqtref.x"(%qubit) : (!mqtref.Qubit) -> ()
"mqtref.x"(%qubit) : (!mqtref.Qubit) -> ()

// Measure the qubit
// CHECK: %bit = "mqtref.measure"(%qubit) : (!mqtref.Qubit) -> i1
// CHECK-GENERIC: %bit = "mqtref.measure"(%qubit) : (!mqtref.Qubit) -> i1
%bit = "mqtref.measure"(%qubit) : (!mqtref.Qubit) -> i1

// Deallocate the qubit
// CHECK: "mqtref.deallocQubit"(%qubit) : (!mqtref.Qubit) -> ()
// CHECK-GENERIC: "mqtref.deallocQubit"(%qubit) : (!mqtref.Qubit) -> ()
"mqtref.deallocQubit"(%qubit) : (!mqtref.Qubit) -> ()
