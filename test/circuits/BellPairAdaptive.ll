; ModuleID = 'Adaptive module implementing Bell-pair correlation via classical correction'
source_filename = "BellPairAdaptive.ll"

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
  %r0 = call %Result* @__quantum__qis__m__body(%Qubit* %q0)
  %b = call i1 @__quantum__rt__read_result(%Result* %r0)
  br i1 %b, label %correct, label %record

correct:
  call void @__quantum__qis__x__body(%Qubit* %q1)
  br label %record

record:
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

declare void @__quantum__qis__x__body(%Qubit*)

declare %Result* @__quantum__qis__m__body(%Qubit*) #1

declare i1 @__quantum__rt__read_result(%Result*)

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
