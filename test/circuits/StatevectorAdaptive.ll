; Adaptive state extraction with dynamic arrays, helpers and terminal measurements.
define i64 @main() #0 {
entry:
  call void @__quantum__rt__initialize(ptr null)
  %qubits = alloca ptr, i64 3
  %results = alloca ptr, i64 3
  call void @__quantum__rt__qubit_array_allocate(i64 3, ptr %qubits, ptr null)
  call void @__quantum__rt__result_array_allocate(i64 3, ptr %results, ptr null)
  call void @prepare(ptr %qubits)
  br label %measure
measure:
  %i = phi i64 [ 0, %entry ], [ %next, %continue ]
  %qp = getelementptr ptr, ptr %qubits, i64 %i
  %q = load ptr, ptr %qp
  %rp = getelementptr ptr, ptr %results, i64 %i
  %r = load ptr, ptr %rp
  call void @measure(ptr %q, ptr %r)
  %bit = call i1 @__quantum__rt__read_result(ptr %r)
  call void @__quantum__rt__bool_record_output(i1 %bit, ptr null)
  %first = icmp eq i64 %i, 0
  br i1 %first, label %independent, label %continue
independent:
  %third = getelementptr ptr, ptr %qubits, i64 2
  %q2 = load ptr, ptr %third
  call void @__quantum__qis__x__body(ptr %q2)
  br label %continue
continue:
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, 3
  br i1 %done, label %exit, label %measure
exit:
  call void @__quantum__rt__qubit_array_release(i64 3, ptr %qubits)
  call void @__quantum__rt__result_array_record_output(i64 3, ptr %results, ptr null)
  call void @__quantum__rt__result_array_release(i64 3, ptr %results)
  %unused = call ptr @__quantum__rt__qubit_allocate(ptr null)
  call void @__quantum__rt__qubit_release(ptr %unused)
  ret i64 0
}
define void @measure(ptr %q, ptr %r) {
  call void @__quantum__qis__mz__body(ptr %q, ptr %r)
  ret void
}
define void @prepare(ptr %qubits) {
  %q0 = load ptr, ptr %qubits
  %second = getelementptr ptr, ptr %qubits, i64 1
  %q1 = load ptr, ptr %second
  call void @__quantum__qis__gphase__body(double 0.3)
  call void @__quantum__qis__h__body(ptr %q0)
  call void @__quantum__qis__cx__body(ptr %q0, ptr %q1)
  ret void
}
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__rt__qubit_array_allocate(i64, ptr, ptr)
declare void @__quantum__rt__result_array_allocate(i64, ptr, ptr)
declare void @__quantum__rt__qubit_array_release(i64, ptr)
declare void @__quantum__rt__result_array_release(i64, ptr)
declare ptr @__quantum__rt__qubit_allocate(ptr)
declare void @__quantum__rt__qubit_release(ptr)
declare void @__quantum__rt__result_array_record_output(i64, ptr, ptr)
declare i1 @__quantum__rt__read_result(ptr)
declare void @__quantum__rt__bool_record_output(i1, ptr)
declare void @__quantum__qis__gphase__body(double)
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__cx__body(ptr, ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"dynamic_qubit_management", i1 true}
!1 = !{i32 1, !"dynamic_result_management", i1 true}
