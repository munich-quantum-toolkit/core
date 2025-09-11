// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: MQTOPT_ROUNDTRIP
// RUN: MQTOPT_GENERIC_ROUNDTRIP

// Test simple qubit allocation and deallocation
// Allocate a single qubit register with size 1
// CHECK: %qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
// CHECK-GENERIC: %qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
%qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

// Deallocate the register
// CHECK: "mqtopt.deallocQubitRegister"(%qreg) : (!mqtopt.QubitRegister) -> ()
// CHECK-GENERIC: "mqtopt.deallocQubitRegister"(%qreg) : (!mqtopt.QubitRegister) -> ()
"mqtopt.deallocQubitRegister"(%qreg) : (!mqtopt.QubitRegister) -> ()
