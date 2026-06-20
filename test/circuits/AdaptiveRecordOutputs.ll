; ModuleID = 'Adaptive module implementing a 3-qubit Hamming weight'
source_filename = "AdaptiveRecordOutputs.ll"

%Qubit = type opaque
%Result = type opaque

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

define i32 @main() #0 {
entry:
  call void @__quantum__rt__initialize(i8* null)
  %q0 = call %Qubit* @__quantum__rt__qubit_allocate()
  %q1 = call %Qubit* @__quantum__rt__qubit_allocate()
  %q2 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__h__body(%Qubit* %q0)
  call void @__quantum__qis__h__body(%Qubit* %q1)
  call void @__quantum__qis__h__body(%Qubit* %q2)
  %r0 = call %Result* @__quantum__qis__m__body(%Qubit* %q0)
  %r1 = call %Result* @__quantum__qis__m__body(%Qubit* %q1)
  %r2 = call %Result* @__quantum__qis__m__body(%Qubit* %q2)
  %b0 = call i1 @__quantum__rt__read_result(%Result* %r0)
  %b1 = call i1 @__quantum__rt__read_result(%Result* %r1)
  %b2 = call i1 @__quantum__rt__read_result(%Result* %r2)

  ; Classical compute: Hamming weight and its mean.
  %c0 = zext i1 %b0 to i64
  %c1 = zext i1 %b1 to i64
  %c2 = zext i1 %b2 to i64
  %sum01 = add i64 %c0, %c1
  %weight = add i64 %sum01, %c2
  %weight_f = sitofp i64 %weight to double
  %num_qubits_f = uitofp i64 3 to double
  %mean_f = fdiv double %weight_f, %num_qubits_f

  call void @__quantum__rt__qubit_release(%Qubit* %q0)
  call void @__quantum__rt__qubit_release(%Qubit* %q1)
  call void @__quantum__rt__qubit_release(%Qubit* %q2)

  ; Record the raw measurement bits (these feed the histogram bucketing key).
  call void @__quantum__rt__result_record_output(%Result* %r0, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @r0_lbl, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* %r1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @r1_lbl, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* %r2, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @r2_lbl, i32 0, i32 0))

  ; Output: tuple of 3 elements (array of 3 bools, int count, float mean).
  call void @__quantum__rt__tuple_record_output(i64 3, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @outputs_lbl, i32 0, i32 0))
  call void @__quantum__rt__array_record_output(i64 3, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @measurements_lbl, i32 0, i32 0))
  call void @__quantum__rt__bool_record_output(i1 %b0, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @m0_lbl, i32 0, i32 0))
  call void @__quantum__rt__bool_record_output(i1 %b1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @m1_lbl, i32 0, i32 0))
  call void @__quantum__rt__bool_record_output(i1 %b2, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @m2_lbl, i32 0, i32 0))
  call void @__quantum__rt__int_record_output(i64 %weight, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @weight_lbl, i32 0, i32 0))
  call void @__quantum__rt__float_record_output(double %mean_f, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @mean_lbl, i32 0, i32 0))

  call void @__quantum__rt__result_update_reference_count(%Result* %r0, i32 -1)
  call void @__quantum__rt__result_update_reference_count(%Result* %r1, i32 -1)
  call void @__quantum__rt__result_update_reference_count(%Result* %r2, i32 -1)
  ret i32 0
}

declare void @__quantum__qis__h__body(%Qubit*)

declare %Result* @__quantum__qis__m__body(%Qubit*) #1

declare i1 @__quantum__rt__read_result(%Result*)

declare void @__quantum__rt__initialize(i8*)

declare %Qubit* @__quantum__rt__qubit_allocate()

declare void @__quantum__rt__qubit_release(%Qubit*)

declare void @__quantum__rt__result_record_output(%Result*, i8*)

declare void @__quantum__rt__tuple_record_output(i64, i8*)

declare void @__quantum__rt__array_record_output(i64, i8*)

declare void @__quantum__rt__bool_record_output(i1, i8*)

declare void @__quantum__rt__int_record_output(i64, i8*)

declare void @__quantum__rt__float_record_output(double, i8*)

declare void @__quantum__rt__result_update_reference_count(%Result*, i32)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="3" "required_num_results"="3" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 true}
!3 = !{i32 1, !"dynamic_result_management", i1 true}
