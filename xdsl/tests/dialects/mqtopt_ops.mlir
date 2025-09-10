// RUN: MQTOPT_ROUNDTRIP
// RUN: MQTOPT_GENERIC_ROUNDTRIP

// Test simple qubit allocation, X gate, and deallocation
// Allocate a single qubit register with size 1
// CHECK: %qreg = mqtopt.allocQubitRegister
// CHECK-GENERIC: %qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
%qreg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

// Create a test qubit (simplified for now)
%q0 = "test.qubit"() : () -> !mqtopt.Qubit

// Apply X gate
// CHECK: %q1 = mqtopt.x %q0
// CHECK-GENERIC: %q1 = "mqtopt.x"(%q0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
%q1 = "mqtopt.x"(%q0) : (!mqtopt.Qubit) -> !mqtopt.Qubit

// Deallocate the register
// CHECK: mqtopt.deallocQubitRegister %qreg
// CHECK-GENERIC: "mqtopt.deallocQubitRegister"(%qreg) : (!mqtopt.QubitRegister) -> ()
"mqtopt.deallocQubitRegister"(%qreg) : (!mqtopt.QubitRegister) -> ()
