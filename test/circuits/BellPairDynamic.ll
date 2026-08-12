; ModuleID = 'Dynamic module implementing Bell pair'
source_filename = "BellPairDynamic.ll"

@0 = internal constant [3 x i8] c"r0\00"
@1 = internal constant [3 x i8] c"r1\00"

define i64 @main() #0 {
entry:
  call void @__quantum__rt__initialize(ptr null)
  %q0 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %q1 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  call void @__quantum__qis__h__body(ptr %q0)
  call void @__quantum__qis__cnot__body(ptr %q0, ptr %q1)
  %r0 = call ptr @__quantum__rt__result_allocate(ptr null)
  %r1 = call ptr @__quantum__rt__result_allocate(ptr null)
  call void @__quantum__qis__mz__body(ptr %q0, ptr %r0)
  call void @__quantum__qis__mz__body(ptr %q1, ptr %r1)
  call void @__quantum__rt__qubit_release(ptr %q0)
  call void @__quantum__rt__qubit_release(ptr %q1)
  call void @__quantum__rt__result_record_output(ptr %r0, ptr @0)
  call void @__quantum__rt__result_record_output(ptr %r1, ptr @1)
  call void @__quantum__rt__result_release(ptr %r0)
  call void @__quantum__rt__result_release(ptr %r1)
  ret i64 0
}

declare void @__quantum__qis__h__body(ptr)

declare void @__quantum__qis__cnot__body(ptr, ptr)

declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1

declare void @__quantum__rt__initialize(ptr)

declare ptr @__quantum__rt__qubit_allocate(ptr)

declare ptr @__quantum__rt__result_allocate(ptr)

declare void @__quantum__rt__qubit_release(ptr)

declare void @__quantum__rt__result_record_output(ptr, ptr)

declare void @__quantum__rt__result_release(ptr)

attributes #0 = { "entry_point" "output_labeling_schema"="labeled" "qir_profiles"="adaptive_profile" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 2}
!1 = !{i32 7, !"qir_minor_version", i32 1}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 true}
