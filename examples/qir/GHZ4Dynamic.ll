; ModuleID = 'bell'
source_filename = "bell"

%Qubit = type opaque
%Result = type opaque
%Array = type opaque

@0 = internal constant [3 x i8] c"r0\00"
@1 = internal constant [3 x i8] c"r1\00"
@2 = internal constant [3 x i8] c"r2\00"
@3 = internal constant [3 x i8] c"r3\00"

define i32 @main() #0 {
entry:
  call void @__quantum__rt__initialize(ptr null)
  %q = call ptr @__quantum__rt__qubit_allocate_array(i64 4)
  %a0 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %q, i64 0)
  %q0 = load ptr, ptr %a0, align 8
  %a1 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %q, i64 1)
  %q1 = load ptr, ptr %a1, align 8
  %a2 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %q, i64 2)
  %q2 = load ptr, ptr %a2, align 8
  %a3 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %q, i64 3)
  %q3 = load ptr, ptr %a3, align 8
  call void @__quantum__qis__h__body(ptr %q0)
  call void @__quantum__qis__cnot__body(ptr %q0, ptr %q1)
  call void @__quantum__qis__cnot__body(ptr %q1, ptr %q2)
  call void @__quantum__qis__cnot__body(ptr %q2, ptr %q3)
  %r0 = call ptr @__quantum__qis__m__body(ptr %q0)
  %r1 = call ptr @__quantum__qis__m__body(ptr %q1)
  %r2 = call ptr @__quantum__qis__m__body(ptr %q2)
  %r3 = call ptr @__quantum__qis__m__body(ptr %q3)
  call void @__quantum__rt__qubit_release_array(ptr %q)
  %r = call ptr @__quantum__rt__array_create_1d(i32 8, i64 4)
  %b0 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %r, i64 0)
  store ptr %r0, ptr %b0, align 8
  %b1 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %r, i64 0)
  store ptr %r1, ptr %b1, align 8
  %b2 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %r, i64 0)
  store ptr %r2, ptr %b2, align 8
  %b3 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %r, i64 0)
  store ptr %r3, ptr %b3, align 8
  %o0 = load ptr, ptr %b0, align 8
  call void @__quantum__rt__result_record_output(ptr %o0, ptr @0)
  %o1 = load ptr, ptr %b1, align 8
  call void @__quantum__rt__result_record_output(ptr %o1, ptr @1)
  %o2 = load ptr, ptr %b2, align 8
  call void @__quantum__rt__result_record_output(ptr %o2, ptr @2)
  %o3 = load ptr, ptr %b3, align 8
  call void @__quantum__rt__result_record_output(ptr %o3, ptr @3)
  call void @__quantum__rt__array_update_reference_count(ptr %r, i32 -1)
  ret i32 0
}

declare void @__quantum__qis__h__body(ptr)

declare void @__quantum__qis__cnot__body(ptr, ptr)

declare ptr @__quantum__qis__m__body(ptr) #1

declare void @__quantum__rt__initialize(ptr)

declare ptr @__quantum__rt__qubit_allocate_array(i64)

declare ptr @__quantum__rt__array_create_1d(i32, i64)

declare ptr @__quantum__rt__array_get_element_ptr_1d(ptr, i64)

declare void @__quantum__rt__qubit_release_array(ptr)

declare void @__quantum__rt__array_update_reference_count(ptr, i32)

declare void @__quantum__rt__result_record_output(ptr, ptr)


attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="4" "required_num_results"="4" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
