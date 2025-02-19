// initially taken from
// https://github.com/qir-alliance/qir-runner/blob/main/stdlib/include/qir_stdlib.h
// and adopted to match the QIR specification
// https://github.com/qir-alliance/qir-spec/tree/main/specification/v0.1

// Instructions to wrap a C++ class with a C interface are taken from
// [https://stackoverflow.com/a/11971205](https://stackoverflow.com/a/11971205)

#pragma once

// NOLINTBEGIN(bugprone-reserved-identifier)
// NOLINTBEGIN(modernize-use-using)
// NOLINTBEGIN(modernize-deprecated-headers)
// NOLINTBEGIN(readability-identifier-naming)

#include <stdint.h>

#ifdef __cplusplus
#define NORETURN [[noreturn]]
extern "C" {
#else
#include <stdnoreturn.h>
#define NORETURN noreturn
#endif

// *** SIMPLE TYPES ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#simple-types
typedef enum : uint8_t {
  PauliI = 0,
  PauliX = 1,
  PauliZ = 2,
  PauliY = 3,
} Pauli;
typedef struct {
  int64_t start;
  int64_t step;
  int64_t end;
} Range;

// moved up, because it is required for Strings already, see BigInt section for
// more
typedef struct BigIntImpl BigInt;

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

// *** STRINGS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#strings

typedef struct StringImpl String;

/// Creates a string from an array of UTF-8 bytes.
String* __quantum__rt__string_create(const char*);

/// Returns a pointer to the zero-terminated array of UTF-8 bytes.
const char* __quantum__rt__string_get_data(const String*);

/// Returns the length of the byte array that contains the string data.
int32_t __quantum__rt__string_get_length(const String*);

/// Adds the given integer value to the reference count for the string.
/// Deallocates the string if the reference count becomes 0.
void __quantum__rt__string_update_reference_count(String*, int32_t);

/// Creates a new string that is the concatenation of the two argument strings.
String* __quantum__rt__string_concatenate(const String*, const String*);

/// Returns true if the two strings are equal, false otherwise.
bool __quantum__rt__string_equal(const String*, const String*);

/// Returns a string representation of the integer.
String* __quantum__rt__int_to_string(int64_t);

/// Returns a string representation of the double.
String* __quantum__rt__double_to_string(double);

/// Returns a string representation of the Boolean.
String* __quantum__rt__bool_to_string(bool);

/// Returns a string representation of the result.
String* __quantum__rt__result_to_string(Result*);

/// Returns a string representation of the Pauli.
String* __quantum__rt__pauli_to_string(Pauli);

/// Returns a string representation of the qubit.
String* __quantum__rt__qubit_to_string(Qubit*);

/// Returns a string representation of the range.
String* __quantum__rt__range_to_string(Range);

/// Returns a string representation of the big integer.
String* __quantum__rt__bigint_to_string(BigInt*);

// *** BIG INTEGERS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#big-integers

/// Creates a big integer with the specified initial value.
BigInt* __quantum__rt__bigint_create_i64(int64_t);

/// Creates a big integer with the initial value specified by the i8 array. The
/// 0-th element of the array is the highest-order byte, followed by the first
/// element, etc.
BigInt* __quantum__rt__bigint_create_array(int32_t, int8_t*);

/// Returns a pointer to the i8 array containing the value of the big integer.
int8_t* __quantum__rt__bigint_get_data(BigInt*);

/// Returns the length of the i8 array that represents the big integer value.
int32_t __quantum__rt__bigint_get_length(BigInt*);

/// Adds the given integer value to the reference count for the big integer.
/// Deallocates the big integer if the reference count becomes 0. The behavior
/// is undefined if the reference count becomes negative.
void __quantum__rt__bigint_update_reference_count(BigInt*, int32_t);

/// Returns the negative of the big integer.
BigInt* __quantum__rt__bigint_negate(BigInt*);

/// Adds two big integers and returns their sum.
BigInt* __quantum__rt__bigint_add(BigInt*, BigInt*);

/// Subtracts the second big integer from the first and returns their
/// difference.
BigInt* __quantum__rt__bigint_subtract(BigInt*, BigInt*);

/// Multiplies two big integers and returns their product.
BigInt* __quantum__rt__bigint_multiply(BigInt*, BigInt*);

/// Divides the first big integer by the second and returns their quotient.
BigInt* __quantum__rt__bigint_divide(BigInt*, BigInt*);

/// Returns the first big integer modulo the second.
BigInt* __quantum__rt__bigint_modulus(BigInt*, BigInt*);

/// Returns the big integer raised to the integer power.
BigInt* __quantum__rt__bigint_power(BigInt*, int32_t);

/// Returns the bitwise-AND of two big integers.
BigInt* __quantum__rt__bigint_bitand(BigInt*, BigInt*);

/// Returns the bitwise-OR of two big integers.
BigInt* __quantum__rt__bigint_bitor(BigInt*, BigInt*);

/// Returns the bitwise-XOR of two big integers.
BigInt* __quantum__rt__bigint_bitxor(BigInt*, BigInt*);

/// Returns the bitwise complement of the big integer.
BigInt* __quantum__rt__bigint_bitnot(BigInt*);

/// Returns the big integer arithmetically shifted left by the (positive)
/// integer amount of bits.
BigInt* __quantum__rt__bigint_shiftleft(BigInt*, int64_t);

/// Returns the big integer arithmetically shifted right by the (positive)
/// integer amount of bits.
BigInt* __quantum__rt__bigint_shiftright(BigInt*, int64_t);

/// Returns true if the two big integers are equal, false otherwise.
bool __quantum__rt__bigint_equal(BigInt*, BigInt*);

/// Returns true if the first big integer is greater than the second, false
/// otherwise.
bool __quantum__rt__bigint_greater(BigInt*, BigInt*);

/// Returns true if the first big integer is greater than or equal to the
/// second, false otherwise.
bool __quantum__rt__bigint_greater_eq(BigInt*, BigInt*);

// *** TUPLES ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#tuples-and-user-defined-types

typedef struct TupleImpl Tuple;

/// Allocates space for a tuple requiring the given number of bytes and sets the
/// reference count to 1.
Tuple* __quantum__rt__tuple_create(int64_t);

/// Creates a shallow copy of the tuple if the user count is larger than 0 or
/// the second argument is `true`.
Tuple* __quantum__rt__tuple_copy(Tuple*, bool force);

/// Adds the given integer value to the reference count for the tuple.
/// Deallocates the tuple if the reference count becomes 0. The behavior is
/// undefined if the reference count becomes negative.
void __quantum__rt__tuple_update_reference_count(Tuple*, int32_t);

/// Adds the given integer value to the alias count for the tuple. Fails if the
/// count becomes negative.
void __quantum__rt__tuple_update_alias_count(Tuple*, int32_t);

// *** ARRAYS ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/1_Data_Types.md#arrays

typedef struct ArrayImpl Array;

/// Creates a new 1-dimensional array. The int64_t is the size of each element
/// in bytes. The int64_t is the length of the array. The bytes of the new array
/// should be set to zero.
Array* __quantum__rt__array_create_1d(int32_t, int64_t);

/// Creates a shallow copy of the array if the user count is larger than 0 or
/// the second argument is `true`.
Array* __quantum__rt__array_copy(Array*, bool);

/// Returns a new array which is the concatenation of the two passed-in arrays.
Array* __quantum__rt__array_concatenate(Array*, Array*);

/// Creates and returns an array that is a slice of an existing 1-dimensional
/// array. The slice may be accessing the same memory as the given array unless
/// its alias count is larger than 0 or the last argument is true. The Range
/// specifies the indices that should be the elements of the returned array.
Array* __quantum__rt__array_slice_1d(Array*, Range, bool);

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

/// Adds the given integer value to the alias count for the array. Fails if the
/// count becomes negative.
void __quantum__rt__array_update_alias_count(Array*, int32_t);

/**
 * Creates a new array. The first i32 is the size of each element in bytes. The
 * second i32 is the dimension count. The i64* should point to an array of i64s
 * contains the length of each dimension. The bytes of the new array should be
 * set to zero. If any length is zero, the result should be an empty array with
 * the given number of dimensions.
 */
Array* __quantum__rt__array_create(int32_t, int32_t, int64_t*);

/// Returns the number of dimensions in the array.
int32_t __quantum__rt__array_get_dim(Array*);

/// Returns the length of a dimension of the array. The i32 is the zero-based
/// dimension to return the length of; it must be smaller than the number of
/// dimensions in the array.
int64_t __quantum__rt__array_get_size(Array*, int32_t);

/// Returns a pointer to the indicated element of the array. The i64* should
/// point to an array of i64s that are the indices for each dimension.
int8_t* __quantum__rt__array_get_element_ptr(Array*, int64_t*);

/**
 * Creates and returns an array that is a slice of an existing array. The slice
 * may be accessing the same memory as the given array unless its alias count is
 * larger than 0 or the last argument is true. The i32 indicates which dimension
 * the slice is on, and must be smaller than the number of dimensions in the
 * array. The %Range specifies the indices in that dimension that should be the
 * elements of the returned array. The reference count of the elements remains
 * unchanged.
 */
Array* __quantum__rt__array_slice(Array*, int32_t, Range, bool);

/**
 * Creates and returns an array that is a projection of an existing array. The
 * projection may be accessing the same memory as the given array unless its
 * alias count is larger than 0 or the last argument is true. The i32 indicates
 * which dimension the projection is on, and the i64 specifies the index in that
 * dimension to project. The reference count of all array elements remains
 * unchanged. If the existing array is one-dimensional then a runtime failure
 * should occur.
 */
Array* __quantum__rt__array_project(Array*, int32_t, int64_t, bool);

// *** CALLABLES ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/2_Callables.md

typedef struct CallablImpl Callable;

/// Initializes the callable with the provided function table and capture tuple.
/// The capture tuple pointer should be null if there is no capture.
Callable* __quantum__rt__callable_create(void (*f[4])(Tuple*, Tuple*, Tuple*),
                                         void (*m[2])(Tuple*, Tuple*, Tuple*),
                                         Tuple*);

/// Creates a shallow copy of the callable if the alias count is larger than 0
/// or the second argument is `true`. Returns the given callable pointer
/// otherwise, after increasing its reference count by 1.
Callable* __quantum__rt__callable_copy(Callable*, bool);

/// Invokes the callable with the provided argument tuple and fills in the
/// result tuple.
void __quantum__rt__callable_invoke(Callable*, Tuple*, Tuple*);

/// Updates the callable by applying the Adjoint functor.
void __quantum__rt__callable_make_adjoint(Callable*);

/// Updates the callable by applying the Controlled functor.
void __quantum__rt__callable_make_controlled(Callable*);

/// Adds the given integer value to the reference count for the callable.
/// Deallocates the callable if the reference count becomes 0. The behavior is
/// undefined if the reference count becomes negative.
void __quantum__rt__callable_update_reference_count(Callable*, int32_t);

/// Adds the given integer value to the alias count for the callable. Fails if
/// the count becomes negative.
void __quantum__rt__callable_update_alias_count(Callable*, int32_t);

/// Invokes the function at index 0 in the memory management table of the
/// callable with the capture tuple and the given 32-bit integer.
void __quantum__rt__capture_update_reference_count(Callable*, int32_t);

/// Invokes the function at index 1 in the memory management table of the
/// callable with the capture tuple and the given 32-bit integer.
void __quantum__rt__capture_update_alias_count(Callable*, int32_t);

// *** CLASSICAL RUNTIME ***
// cf.
// https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/3_Classical_Runtime.md

/// Include the given message in the computation's execution log or equivalent.
void __quantum__rt__message(const String* msg);

/// Fail the computation with the given error message.
NORETURN void __quantum__rt__fail(const String* msg);

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
void __quantum__qis__sqrtx__body(Qubit*);
void __quantum__qis__sqrtxdg__body(Qubit*);
void __quantum__qis__t__body(Qubit*);
void __quantum__qis__tdg__body(Qubit*);
void __quantum__qis__rx__body(double, Qubit*);
void __quantum__qis__ry__body(double, Qubit*);
void __quantum__qis__rz__body(double, Qubit*);
void __quantum__qis__p__body(double, Qubit*);
void __quantum__qis__u__body(double, double, double, Qubit*);
void __quantum__qis__u3__body(double, double, double, Qubit*);
void __quantum__qis__u2__body(double, double, Qubit*);
void __quantum__qis__u1__body(double, Qubit*);
void __quantum__qis__cu1__body(double, Qubit*, Qubit*);
void __quantum__qis__cnot__body(Qubit*, Qubit*);
void __quantum__qis__cx__body(Qubit*, Qubit*);
void __quantum__qis__cy__body(Qubit*, Qubit*);
void __quantum__qis__cz__body(Qubit*, Qubit*);
void __quantum__qis__swap__body(Qubit*, Qubit*);
void __quantum__qis__cswap__body(Qubit*, Qubit*, Qubit*);
void __quantum__qis__crz__body(double, Qubit*, Qubit*);
void __quantum__qis__cry__body(double, Qubit*, Qubit*);
void __quantum__qis__cp__body(double, Qubit*, Qubit*);
void __quantum__qis__rzz__body(double, Qubit*, Qubit*);
void __quantum__qis__ccx__body(Qubit*, Qubit*, Qubit*);
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

/**
 * Inserts a marker in the generated output that indicates the start of a tuple
 * and how many tuple elements it has. The second parameter defines a string
 * label for the tuple. Depending on the output schema, the label is included
 * in the output or omitted.
 */
void __quantum__rt__tuple_record_output(int64_t, const char*);

/**
 * Inserts a marker in the generated output that indicates the start of an
 * array and how many array elements it has. The second parameter defines a
 * string label for the array. Depending on the output schema, the label is
 * included in the output or omitted.
 */
void __quantum__rt__array_record_output(int64_t, const char*);

/// Adds a measurement result to the generated output. The second parameter
/// defines a string label for the result value. Depending on the output schema,
/// the label is included in the output or omitted.
void __quantum__rt__result_record_output(Result*, const char*);

/// Adds a boolean value to the generated output. The second parameter defines
/// a string label for the result value. Depending on the output schema, the
/// label is included in the output or omitted.
void __quantum__rt__bool_record_output(bool, const char*);

// NOLINTEND(readability-identifier-naming)
// NOLINTEND(modernize-deprecated-headers)
// NOLINTEND(modernize-use-using)
// NOLINTEND(bugprone-reserved-identifier)

#ifdef __cplusplus
} // extern "C"
#endif
