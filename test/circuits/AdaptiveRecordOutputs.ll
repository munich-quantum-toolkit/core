; ModuleID = 'Adaptive module implementing a 3-qubit Hamming weight'
source_filename = "AdaptiveRecordOutputs.ll"

@r0_lbl = internal constant [3 x i8] c"r0\00"
@r1_lbl = internal constant [3 x i8] c"r1\00"
@r2_lbl = internal constant [3 x i8] c"r2\00"
@outputs_lbl = internal constant [8 x i8] c"outputs\00"
@measurements_lbl = internal constant [15 x i8] c"  measurements\00"
@m0_lbl = internal constant [7 x i8] c"    m0\00"
@m1_lbl = internal constant [7 x i8] c"    m1\00"
@m2_lbl = internal constant [7 x i8] c"    m2\00"
@weight_lbl = internal constant [17 x i8] c"  hamming_weight\00"
@mean_lbl = internal constant [7 x i8] c"  mean\00"

define i64 @main() #0 {
entry:
  call void @__quantum__rt__initialize(ptr null)
  %q0 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %q1 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %q2 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  call void @__quantum__qis__h__body(ptr %q0)
  call void @__quantum__qis__h__body(ptr %q1)
  call void @__quantum__qis__h__body(ptr %q2)
  %r0 = call ptr @__quantum__rt__result_allocate(ptr null)
  %r1 = call ptr @__quantum__rt__result_allocate(ptr null)
  %r2 = call ptr @__quantum__rt__result_allocate(ptr null)
  call void @__quantum__qis__mz__body(ptr %q0, ptr %r0)
  call void @__quantum__qis__mz__body(ptr %q1, ptr %r1)
  call void @__quantum__qis__mz__body(ptr %q2, ptr %r2)
  %b0 = call i1 @__quantum__rt__read_result(ptr %r0)
  %b1 = call i1 @__quantum__rt__read_result(ptr %r1)
  %b2 = call i1 @__quantum__rt__read_result(ptr %r2)

  ; Classical compute: Hamming weight and its mean.
  %c0 = zext i1 %b0 to i64
  %c1 = zext i1 %b1 to i64
  %c2 = zext i1 %b2 to i64
  %sum01 = add i64 %c0, %c1
  %weight = add i64 %sum01, %c2
  %weight_f = sitofp i64 %weight to double
  %num_qubits_f = uitofp i64 3 to double
  %mean_f = fdiv double %weight_f, %num_qubits_f

  call void @__quantum__rt__qubit_release(ptr %q0)
  call void @__quantum__rt__qubit_release(ptr %q1)
  call void @__quantum__rt__qubit_release(ptr %q2)

  ; Record the raw measurement bits (these feed the histogram bucketing key).
  call void @__quantum__rt__result_record_output(ptr %r0, ptr @r0_lbl)
  call void @__quantum__rt__result_record_output(ptr %r1, ptr @r1_lbl)
  call void @__quantum__rt__result_record_output(ptr %r2, ptr @r2_lbl)

  ; Output: tuple of 3 elements (array of 3 bools, int count, double mean).
  call void @__quantum__rt__tuple_record_output(i64 3, ptr @outputs_lbl)
  call void @__quantum__rt__array_record_output(i64 3, ptr @measurements_lbl)
  call void @__quantum__rt__bool_record_output(i1 %b0, ptr @m0_lbl)
  call void @__quantum__rt__bool_record_output(i1 %b1, ptr @m1_lbl)
  call void @__quantum__rt__bool_record_output(i1 %b2, ptr @m2_lbl)
  call void @__quantum__rt__int_record_output(i64 %weight, ptr @weight_lbl)
  call void @__quantum__rt__double_record_output(double %mean_f, ptr @mean_lbl)

  call void @__quantum__rt__result_release(ptr %r0)
  call void @__quantum__rt__result_release(ptr %r1)
  call void @__quantum__rt__result_release(ptr %r2)
  ret i64 0
}

declare void @__quantum__qis__h__body(ptr)

declare void @__quantum__qis__mz__body(ptr, ptr writeonly) #1

declare i1 @__quantum__rt__read_result(ptr)

declare void @__quantum__rt__initialize(ptr)

declare ptr @__quantum__rt__qubit_allocate(ptr)

declare ptr @__quantum__rt__result_allocate(ptr)

declare void @__quantum__rt__qubit_release(ptr)

declare void @__quantum__rt__result_record_output(ptr, ptr)

declare void @__quantum__rt__tuple_record_output(i64, ptr)

declare void @__quantum__rt__array_record_output(i64, ptr)

declare void @__quantum__rt__bool_record_output(i1, ptr)

declare void @__quantum__rt__int_record_output(i64, ptr)

declare void @__quantum__rt__double_record_output(double, ptr)

declare void @__quantum__rt__result_release(ptr)

attributes #0 = { "entry_point" "output_labeling_schema"="labeled" "qir_profiles"="adaptive_profile" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 2}
!1 = !{i32 7, !"qir_minor_version", i32 1}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 true}
