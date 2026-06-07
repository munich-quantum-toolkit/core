/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

namespace qdmi_test {

inline constexpr auto QASM3_BELL_SAMPLING = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
)";

inline constexpr auto QASM3_BELL_STATE = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
)";

inline constexpr const char* QASM3_MALFORMED = "Definitely not OpenQASM";

// A slightly heavier 5-qubit sampling circuit to prolong runtime slightly while
// remaining fast
inline constexpr auto QASM3_HEAVY_SAMPLING5 = R"(
OPENQASM 3;
include "stdgates.inc";
qubit[5] q;
bit[5] c;
// GHZ-like entanglement chain
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
// Some single-qubit rotations for additional depth
rx(0.7) q[0];
ry(0.5) q[1];
rz(1.1) q[2];
ry(0.3) q[3];
rx(0.9) q[4];
// Reverse entanglement to add more two-qubit layers
cx q[3], q[4];
cx q[2], q[3];
cx q[1], q[2];
cx q[0], q[1];
// Measure all qubits
c = measure q;
)";

inline constexpr auto QIR_BELL_PAIR_STATIC = R"(
; ModuleID = 'Static module implementing Bell pair'
source_filename = "BellPairStatic.ll"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [3 x i8] c"r0\00"
@1 = internal constant [3 x i8] c"r1\00"

define i32 @main() #0 {
entry:
  call void @__quantum__rt__initialize(i8* null)
  call void @__quantum__qis__h__body(%Qubit* null)
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))
  call void @__quantum__rt__result_record_output(%Result* null, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0))
  ret i32 0
}

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) #1

declare void @__quantum__rt__initialize(i8*)

declare void @__quantum__rt__result_record_output(%Result*, i8*)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="2" "required_num_results"="2" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
)";

inline constexpr auto QIR_BELL_PAIR_DYNAMIC = R"(
; ModuleID = 'Dynamic module implementing Bell pair'
source_filename = "BellPairDynamic.ll"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [3 x i8] c"r0\00"
@1 = internal constant [3 x i8] c"r1\00"

define i32 @main() #0 {
entry:
  call void @__quantum__rt__initialize(i8* null)
  %q0 = call %Qubit* @__quantum__rt__qubit_allocate()
  %q1 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__h__body(%Qubit* %q0)
  call void @__quantum__qis__cnot__body(%Qubit* %q0, %Qubit* %q1)
  %r0 = call %Result* @__quantum__qis__m__body(%Qubit* %q0)
  %r1 = call %Result* @__quantum__qis__m__body(%Qubit* %q1)
  call void @__quantum__rt__qubit_release(%Qubit* %q0)
  call void @__quantum__rt__qubit_release(%Qubit* %q1)
  call void @__quantum__rt__result_record_output(%Result* %r0, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* %r1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0))
  call void @__quantum__rt__result_update_reference_count(%Result* %r0, i32 -1)
  call void @__quantum__rt__result_update_reference_count(%Result* %r1, i32 -1)
  ret i32 0
}

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare %Result* @__quantum__qis__m__body(%Qubit*) #1

declare void @__quantum__rt__initialize(i8*)

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__rt__qubit_release(%Qubit*)

declare void @__quantum__rt__result_record_output(%Result*, i8*)

declare void @__quantum__rt__result_update_reference_count(%Result*, i32)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="2" "required_num_results"="2" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 true}
)";

} // namespace qdmi_test
