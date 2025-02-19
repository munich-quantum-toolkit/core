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
  call void @__quantum__rt__initialize(i8* null)
  %q = call %Array* @__quantum__rt__qubit_allocate_array(i64 4)
  %a0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  %q0 = load %Qubit*, i8* %a0, align 8
  %a1 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 1)
  %q1 = load %Qubit*, i8* %a1, align 8
  %a2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 2)
  %q2 = load %Qubit*, i8* %a2, align 8
  %a3 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 3)
  %q3 = load %Qubit*, i8* %a3, align 8
  call void @__quantum__qis__h__body(%Qubit* %q0)
  call void @__quantum__qis__cnot__body(%Qubit* %q0, %Qubit* %q1)
  call void @__quantum__qis__cnot__body(%Qubit* %q1, %Qubit* %q2)
  call void @__quantum__qis__cnot__body(%Qubit* %q2, %Qubit* %q3)
  %r0 = call %Result* @__quantum__qis__m__body(%Qubit* %q0)
  %r1 = call %Result* @__quantum__qis__m__body(%Qubit* %q1)
  %r2 = call %Result* @__quantum__qis__m__body(%Qubit* %q2)
  %r3 = call %Result* @__quantum__qis__m__body(%Qubit* %q3)
  call void @__quantum__rt__qubit_release_array(%Array* %q)
  call void @__quantum__rt__result_record_output(%Result* %r0, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* %r1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* %r2, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @2, i32 0, i32 0))
  call void @__quantum__rt__result_record_output(%Result* %r3, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @3, i32 0, i32 0))
  ret i32 0
}

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare %Result* @__quantum__qis__m__body(%Qubit*) #1

declare void @__quantum__rt__initialize(i8*)

declare %Array* @__quantum__rt__qubit_allocate_array(i64)

declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64)

declare void @__quantum__rt__qubit_release_array(%Array*)

declare void @__quantum__rt__result_record_output(%Result*, i8*)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="2" "required_num_results"="2" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
