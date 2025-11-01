/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// initially taken from
// https://github.com/qir-alliance/qir-runner/blob/main/stdlib/include/qir_stdlib.h
// and adopted to match the QIR specification
// https://github.com/qir-alliance/qir-spec/tree/main/specification/v0.1

// Instructions to wrap a C++ class with a C interface are taken from
// https://stackoverflow.com/a/11971205

#pragma once

// NOLINTBEGIN(modernize-use-using)
// NOLINTBEGIN(modernize-deprecated-headers)
// NOLINTBEGIN(readability-identifier-naming)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// *** MEASUREMENT RESULTS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#measurement-results

typedef struct ResultImpl Result;

/// Returns a constant representing a measurement result zero.
Result* __quantum__rt__result_get_zero();

/// Returns a constant representing a measurement result one.
Result* __quantum__rt__result_get_one();

/// Returns true if the two results are the same, and false if they are
/// different.
bool __quantum__rt__result_equal(Result*, Result*);

/// Adds the given integer value to the reference count for the result.
/// Deallocates the result if the reference count becomes 0.
void __quantum__rt__result_update_reference_count(Result*, int32_t);

// *** QUBITS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#qubits
typedef struct QubitImpl Qubit;

// *** ARRAYS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#arrays

typedef struct ArrayImpl Array;

/// Creates a new 1-dimensional array. The int64_t is the size of each element
/// in bytes. The int64_t is the length of the array. The bytes of the new array
/// should be set to zero.
Array* __quantum__rt__array_create_1d(int32_t, int64_t);

/// Returns the length of a dimension of the array. The int64_t is the
/// zero-based dimension to return the length of; it must be 0 for a
/// 1-dimensional array.
int64_t __quantum__rt__array_get_size_1d(const Array*);

/// Returns a pointer to the element of the array at the zero-based index given
/// by the int64_t.
int8_t* __quantum__rt__array_get_element_ptr_1d(Array*, int64_t);

/// Adds the given integer value to the reference count for the array.
/// Deallocates the array if the reference count becomes 0. The behavior is
/// undefined if the reference count becomes negative.
void __quantum__rt__array_update_reference_count(Array*, int32_t);

// *** QUANTUM INSTRUCTIONSET AND RUNTIME ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/4_Quantum_Runtime.md

/// Allocates a single qubit.
Qubit* __quantum__rt__qubit_allocate();

/// Creates an array of the given size and populates it with newly-allocated
/// qubits.
Array* __quantum__rt__qubit_allocate_array(int64_t);

/// Releases a single qubit. Passing a null pointer as argument should cause a
/// runtime failure.
void __quantum__rt__qubit_release(Qubit*);

/// Releases an array of qubits; each qubit in the array is released, and the
/// array itself is unreferenced. Passing a null pointer as argument should
/// cause a runtime failure.
void __quantum__rt__qubit_release_array(Array*);

// QUANTUM INSTRUCTION SET
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md#base-profile
// WARNING: This refers to the unstable version of the specification under
// developments.

void __quantum__qis__x__body(Qubit*);
void __quantum__qis__y__body(Qubit*);
void __quantum__qis__z__body(Qubit*);
void __quantum__qis__h__body(Qubit*);
void __quantum__qis__s__body(Qubit*);
void __quantum__qis__sdg__body(Qubit*);
void __quantum__qis__sx__body(Qubit*);
void __quantum__qis__sxdg__body(Qubit*);
void __quantum__qis__sqrtx__body(Qubit*);
void __quantum__qis__sqrtxdg__body(Qubit*);
void __quantum__qis__t__body(Qubit*);
void __quantum__qis__tdg__body(Qubit*);
void __quantum__qis__r__body(Qubit*, double, double);
void __quantum__qis__prx__body(Qubit*, double, double);
void __quantum__qis__rx__body(Qubit*, double);
void __quantum__qis__ry__body(Qubit*, double);
void __quantum__qis__rz__body(Qubit*, double);
void __quantum__qis__p__body(Qubit*, double);
void __quantum__qis__rxx__body(Qubit*, Qubit*, double);
void __quantum__qis__ryy__body(Qubit*, Qubit*, double);
void __quantum__qis__rzz__body(Qubit*, Qubit*, double);
void __quantum__qis__rzx__body(Qubit*, Qubit*, double);
void __quantum__qis__u__body(Qubit*, double, double, double);
void __quantum__qis__u3__body(Qubit*, double, double, double);
void __quantum__qis__u2__body(Qubit*, double, double);
void __quantum__qis__u1__body(Qubit*, double);
void __quantum__qis__cu1__body(Qubit*, Qubit*, double);
void __quantum__qis__cu3__body(Qubit*, Qubit*, double, double, double);
void __quantum__qis__cnot__body(Qubit*, Qubit*);
void __quantum__qis__cx__body(Qubit*, Qubit*);
void __quantum__qis__cy__body(Qubit*, Qubit*);
void __quantum__qis__cz__body(Qubit*, Qubit*);
void __quantum__qis__ch__body(Qubit*, Qubit*);
void __quantum__qis__swap__body(Qubit*, Qubit*);
void __quantum__qis__cswap__body(Qubit*, Qubit*, Qubit*);
void __quantum__qis__crx__body(Qubit*, Qubit*, double);
void __quantum__qis__cry__body(Qubit*, Qubit*, double);
void __quantum__qis__crz__body(Qubit*, Qubit*, double);
void __quantum__qis__cp__body(Qubit*, Qubit*, double);
void __quantum__qis__ccx__body(Qubit*, Qubit*, Qubit*);
void __quantum__qis__ccy__body(Qubit*, Qubit*, Qubit*);
void __quantum__qis__ccz__body(Qubit*, Qubit*, Qubit*);
Result* __quantum__qis__m__body(Qubit*);
Result* __quantum__qis__measure__body(Qubit*);
void __quantum__qis__mz__body(Qubit*, Result*);
void __quantum__qis__reset__body(Qubit*);

// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Adaptive_Profile.md#runtime-functions

/// Initializes the execution environment. Sets all qubits to a zero-state if
/// they are not dynamically managed.
void __quantum__rt__initialize(char*);

/// Reads the value of the given measurement result and converts it to a boolean
/// value.
bool __quantum__rt__read_result(Result*);

/// Adds a measurement result to the generated output. The second parameter
/// defines a string label for the result value. Depending on the output schema,
/// the label is included in the output or omitted.
void __quantum__rt__result_record_output(Result*, const char*);

// NOLINTEND(readability-identifier-naming)
// NOLINTEND(modernize-deprecated-headers)
// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
} // extern "C"
#endif
