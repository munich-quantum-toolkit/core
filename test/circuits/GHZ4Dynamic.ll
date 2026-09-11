; ModuleID = 'ghz4-dynamic'
source_filename = "GHZ4Dynamic.ll"

@results_label = internal constant [8 x i8] c"results\00"

define i64 @main() #0 {
entry:
  %qubits = alloca [4 x ptr], align 8
  %results = alloca [4 x ptr], align 8
  call void @__quantum__rt__initialize(ptr null)
  call void @__quantum__rt__qubit_array_allocate(i64 4, ptr %qubits, ptr null)
  call void @__quantum__rt__result_array_allocate(i64 4, ptr %results, ptr null)

  %q0.slot = getelementptr inbounds [4 x ptr], ptr %qubits, i64 0, i64 0
  %q1.slot = getelementptr inbounds [4 x ptr], ptr %qubits, i64 0, i64 1
  %q2.slot = getelementptr inbounds [4 x ptr], ptr %qubits, i64 0, i64 2
  %q3.slot = getelementptr inbounds [4 x ptr], ptr %qubits, i64 0, i64 3
  %q0 = load ptr, ptr %q0.slot, align 8
  %q1 = load ptr, ptr %q1.slot, align 8
  %q2 = load ptr, ptr %q2.slot, align 8
  %q3 = load ptr, ptr %q3.slot, align 8

  call void @__quantum__qis__h__body(ptr %q0)
  call void @__quantum__qis__cnot__body(ptr %q0, ptr %q1)
  call void @__quantum__qis__cnot__body(ptr %q1, ptr %q2)
  call void @__quantum__qis__cnot__body(ptr %q2, ptr %q3)

  %r0.slot = getelementptr inbounds [4 x ptr], ptr %results, i64 0, i64 0
  %r1.slot = getelementptr inbounds [4 x ptr], ptr %results, i64 0, i64 1
  %r2.slot = getelementptr inbounds [4 x ptr], ptr %results, i64 0, i64 2
  %r3.slot = getelementptr inbounds [4 x ptr], ptr %results, i64 0, i64 3
  %r0 = load ptr, ptr %r0.slot, align 8
  %r1 = load ptr, ptr %r1.slot, align 8
  %r2 = load ptr, ptr %r2.slot, align 8
  %r3 = load ptr, ptr %r3.slot, align 8
  call void @__quantum__qis__mz__body(ptr %q0, ptr %r0)
  call void @__quantum__qis__mz__body(ptr %q1, ptr %r1)
  call void @__quantum__qis__mz__body(ptr %q2, ptr %r2)
  call void @__quantum__qis__mz__body(ptr %q3, ptr %r3)

  call void @__quantum__rt__result_array_record_output(i64 4, ptr %results, ptr @results_label)
  call void @__quantum__rt__result_array_release(i64 4, ptr %results)
  call void @__quantum__rt__qubit_array_release(i64 4, ptr %qubits)
  ret i64 0
}

declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__rt__qubit_array_allocate(i64, ptr, ptr)
declare void @__quantum__rt__qubit_array_release(i64, ptr)
declare void @__quantum__rt__result_array_allocate(i64, ptr, ptr)
declare void @__quantum__rt__result_array_release(i64, ptr)
declare void @__quantum__rt__result_array_record_output(i64, ptr, ptr)
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__cnot__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1

attributes #0 = { "entry_point" "output_labeling_schema"="labeled" "qir_profiles"="adaptive_profile" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"qir_major_version", i32 2}
!1 = !{i32 7, !"qir_minor_version", i32 1}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 true}
!4 = !{i32 1, !"arrays", i1 true}
