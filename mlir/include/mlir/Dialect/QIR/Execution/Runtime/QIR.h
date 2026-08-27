/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file QIR.h
 * @brief C QIR runtime declarations.
 */

// Implements the QIR 2.1 Base and Adaptive Profile runtime surface used by MQT
// Core. The Array and Tuple declarations support the generic controlled
// specializations and are distinct from QIR 2.1 resource arrays.

// Instructions to wrap a C++ class with a C interface are taken from
// https://stackoverflow.com/a/11971205

#pragma once

// NOLINTBEGIN(modernize-use-using)
// NOLINTBEGIN(readability-identifier-naming)

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// *** MEASUREMENT RESULTS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#measurement-results

typedef struct ResultImpl Result;

// *** QUBITS ***
// See
// https://github.com/qir-alliance/qir-spec/blob/main/specification/Instruction_Set.md
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
/// by the int64_t. Returns nullptr if the index is out of bounds.
int8_t* __quantum__rt__array_get_element_ptr_1d(Array*, int64_t);

/// Adds the given integer value to the reference count for the array.
/// Deallocates the array if the reference count becomes 0. The behavior is
/// undefined if the reference count becomes negative.
void __quantum__rt__array_update_reference_count(Array*, int32_t);

// *** TUPLES ***
typedef struct TupleImpl Tuple;

/// Allocate a zero-initialized, suitably aligned tuple payload.
Tuple* __quantum__rt__tuple_create(int64_t size);
void __quantum__rt__tuple_update_reference_count(Tuple*, int32_t);

// *** QUANTUM INSTRUCTIONSET AND RUNTIME ***
// See
// https://github.com/qir-alliance/qir-spec/blob/main/specification/Memory_Management.md

/// Allocate a single qubit using the QIR 2.1 error-pointer convention.
Qubit* __quantum__rt__qubit_allocate(bool* outError);

/// Allocate and release QIR 2.1 caller-owned arrays of qubit pointers.
void __quantum__rt__qubit_array_allocate(int64_t, Qubit**, bool* outError);
void __quantum__rt__qubit_array_release(int64_t, Qubit**);

/// Allocate and release dynamically managed QIR 2.1 results.
Result* __quantum__rt__result_allocate(bool* outError);
void __quantum__rt__result_release(Result*);
void __quantum__rt__result_array_allocate(int64_t, Result**, bool* outError);
void __quantum__rt__result_array_release(int64_t, Result**);

/// Releases a single qubit. Passing a null pointer as argument should cause a
/// runtime failure.
void __quantum__rt__qubit_release(Qubit*);

// QUANTUM INSTRUCTION SET
//
// Dedicated body, adjoint, c*, and cc* functions take parameters first,
// followed by control and target qubits. Generic controlled specializations
// use an Array of controls and the original gate arguments:
// a single target directly, otherwise a Tuple of parameters then targets.

#define MQT_QIR_DECLARE_1_0(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit*);                             \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit*, Qubit*);                    \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit*, Qubit*, Qubit*);           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Qubit*);
#define MQT_QIR_DECLARE_1_1(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(double, Qubit*);                     \
  void __quantum__qis__c##NAME##__##SUFFIX(double, Qubit*, Qubit*);            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double, Qubit*, Qubit*, Qubit*);   \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);
#define MQT_QIR_DECLARE_1_2(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(double, double, Qubit*);             \
  void __quantum__qis__c##NAME##__##SUFFIX(double, double, Qubit*, Qubit*);    \
  void __quantum__qis__cc##NAME##__##SUFFIX(double, double, Qubit*, Qubit*,    \
                                            Qubit*);                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);
#define MQT_QIR_DECLARE_1_3(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(double, double, double, Qubit*);     \
  void __quantum__qis__c##NAME##__##SUFFIX(double, double, double, Qubit*,     \
                                           Qubit*);                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double, double, double, Qubit*,    \
                                            Qubit*, Qubit*);                   \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);
#define MQT_QIR_DECLARE_2_0(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit*, Qubit*);                     \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit*, Qubit*, Qubit*);            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit*, Qubit*, Qubit*, Qubit*);   \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);
#define MQT_QIR_DECLARE_2_1(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(double, Qubit*, Qubit*);             \
  void __quantum__qis__c##NAME##__##SUFFIX(double, Qubit*, Qubit*, Qubit*);    \
  void __quantum__qis__cc##NAME##__##SUFFIX(double, Qubit*, Qubit*, Qubit*,    \
                                            Qubit*);                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);
#define MQT_QIR_DECLARE_2_2(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(double, double, Qubit*, Qubit*);     \
  void __quantum__qis__c##NAME##__##SUFFIX(double, double, Qubit*, Qubit*,     \
                                           Qubit*);                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double, double, Qubit*, Qubit*,    \
                                            Qubit*, Qubit*);                   \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);
#define MQT_QIR_DECLARE_3_0(NAME, SUFFIX, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit*, Qubit*, Qubit*);             \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit*, Qubit*, Qubit*, Qubit*);    \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit*, Qubit*, Qubit*, Qubit*,    \
                                            Qubit*);                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array*, Tuple*);

#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  MQT_QIR_DECLARE_##TARGETS##_##PARAMS(NAME, SUFFIX, CTL_SUFFIX)
#include "mlir/Conversion/GateTable.def"

#undef MQT_QIR_DECLARE_1_0
#undef MQT_QIR_DECLARE_1_1
#undef MQT_QIR_DECLARE_1_2
#undef MQT_QIR_DECLARE_1_3
#undef MQT_QIR_DECLARE_2_0
#undef MQT_QIR_DECLARE_2_1
#undef MQT_QIR_DECLARE_2_2
#undef MQT_QIR_DECLARE_3_0

/// Apply an arbitrary global phase.
void __quantum__qis__gphase__body(double);

/// Common QIS spelling for controlled X.
void __quantum__qis__cnot__body(Qubit*, Qubit*);

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

/// Adds a boolean value to the generated output. The second parameter defines
/// a string label for the value. Depending on the output schema, the label is
/// included in the output or omitted.
void __quantum__rt__bool_record_output(bool, const char*);

/// Adds an integer value to the generated output. The second parameter defines
/// a string label for the value. Depending on the output schema, the label is
/// included in the output or omitted.
void __quantum__rt__int_record_output(int64_t, const char*);

/// Adds a floating-point value to the generated output. The second parameter
/// defines a string label for the value. Depending on the output schema, the
/// label is included in the output or omitted.
void __quantum__rt__double_record_output(double, const char*);

/// Inserts a marker in the generated output indicating that the next
/// \p elementCount recorded values form the contents of a tuple. The second
/// parameter defines a string label for the tuple.
void __quantum__rt__tuple_record_output(int64_t elementCount, const char*);

/// Inserts a marker in the generated output indicating that the next \p size
/// recorded values form the contents of an array. The second parameter defines
/// a string label for the array.
void __quantum__rt__array_record_output(int64_t size, const char*);

/// Record the contents of a caller-owned result pointer array.
void __quantum__rt__result_array_record_output(int64_t, Result**, const char*);

// NOLINTEND(readability-identifier-naming)
// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
} // extern "C"
#endif
