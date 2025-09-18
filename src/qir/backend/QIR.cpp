/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/backend/QIR.h"

#include "ir/operations/OpType.hpp"
#include "qir/backend/Backend.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {

// *** MEASUREMENT RESULTS ***
Result* __quantum__rt__result_get_zero() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Result*>(qir::Backend::RESULT_ZERO_ADDRESS);
}

Result* __quantum__rt__result_get_one() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Result*>(qir::Backend::RESULT_ONE_ADDRESS);
}

bool __quantum__rt__result_equal(Result* result1, Result* result2) {
  auto& backend = qir::Backend::getInstance();
  return backend.equal(result1, result2);
}

void __quantum__rt__result_update_reference_count(Result* result,
                                                  const int32_t k) {
  auto& backend = qir::Backend::getInstance();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  if (result != nullptr &&
      result != reinterpret_cast<Result*>(qir::Backend::RESULT_ZERO_ADDRESS) &&
      result != reinterpret_cast<Result*>(qir::Backend::RESULT_ONE_ADDRESS)) {
    // NOLINTEND(performance-no-int-to-ptr)
    auto& refcount = backend.deref(result).refcount;
    refcount += k;
    if (refcount == 0) {
      backend.rFree(result);
    }
  }
}

// *** STRINGS ***
String* __quantum__rt__string_create(const char* utf8) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* string = new String;
  string->refcount = 1;
  string->content = std::string(utf8);
  return string;
}

const char* __quantum__rt__string_get_data(const String* string) {
  if (string == nullptr) {
    throw std::invalid_argument("The argument must not be null.");
  }
  return string->content.c_str();
}

int32_t __quantum__rt__string_get_length(const String* string) {
  if (string == nullptr) {
    throw std::invalid_argument("The argument must not be null.");
  }
  return static_cast<int32_t>(string->content.size());
}

void __quantum__rt__string_update_reference_count(String* string,
                                                  const int32_t k) {
  if (string != nullptr) {
    string->refcount += k;
    if (string->refcount == 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete string;
    }
  }
}

String* __quantum__rt__string_concatenate(const String* string1,
                                          const String* string2) {
  if (string1 == nullptr) {
    throw std::invalid_argument("The first argument must not be null.");
  }
  if (string2 == nullptr) {
    throw std::invalid_argument("The second argument must not be null.");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* string = new String;
  string->refcount = 1;
  string->content = string1->content + string2->content;
  return string;
}

bool __quantum__rt__string_equal(const String* string1, const String* string2) {
  if (string1 == nullptr) {
    throw std::invalid_argument("The first argument must not be null.");
  }
  if (string2 == nullptr) {
    throw std::invalid_argument("The second argument must not be null.");
  }
  return string1->content == string2->content;
}

String* __quantum__rt__int_to_string(const int64_t z) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* string = new String;
  string->refcount = 1;
  string->content = std::to_string(z);
  return string;
}

String* __quantum__rt__double_to_string(const double d) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* string = new String;
  string->refcount = 1;
  string->content = std::to_string(d);
  return string;
}

String* __quantum__rt__bool_to_string(const bool b) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* string = new String;
  string->refcount = 1;
  string->content = b ? "true" : "false";
  return string;
}

String* __quantum__rt__result_to_string(Result* result) {
  if (result == nullptr) {
    throw std::invalid_argument("The argument must not be null.");
  }
  return __quantum__rt__int_to_string(
      static_cast<int32_t>(__quantum__rt__read_result(result)));
}

String* __quantum__rt__pauli_to_string(const Pauli p) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* string = new String;
  string->refcount = 1;
  switch (p) {
  case PauliI:
    string->content = "PauliI";
    break;
  case PauliX:
    string->content = "PauliX";
    break;
  case PauliZ:
    string->content = "PauliZ";
    break;
  case PauliY:
    string->content = "PauliY";
    break;
  }
  return string;
}

String* __quantum__rt__qubit_to_string(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  return __quantum__rt__int_to_string(
      static_cast<int32_t>(backend.translateAddresses<1>({{qubit}}).front()));
}

String* __quantum__rt__range_to_string(Range /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

String* __quantum__rt__bigint_to_string(BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

// *** BIG INTEGERS ***
BigInt* __quantum__rt__bigint_create_i64(int64_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_create_array(int32_t /*unused*/,
                                           int8_t* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

int8_t* __quantum__rt__bigint_get_data(BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

int32_t __quantum__rt__bigint_get_length(BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__bigint_update_reference_count(BigInt* /*unused*/,
                                                  int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_negate(BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_add(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_subtract(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_multiply(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_divide(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_modulus(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_power(BigInt* /*unused*/, int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_bitand(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_bitor(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_bitxor(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_bitnot(BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_shiftleft(BigInt* /*unused*/,
                                        int64_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

BigInt* __quantum__rt__bigint_shiftright(BigInt* /*unused*/,
                                         int64_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

bool __quantum__rt__bigint_equal(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

bool __quantum__rt__bigint_greater(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

bool __quantum__rt__bigint_greater_eq(BigInt* /*unused*/, BigInt* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

// *** TUPLES ***
Tuple* __quantum__rt__tuple_create(const int64_t size) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* tuple = new Tuple;
  tuple->refcount = 1;
  tuple->aliasCount = 0;
  tuple->data = std::vector<int8_t>(static_cast<size_t>(size), 0);
  return tuple;
}

Tuple* __quantum__rt__tuple_copy(Tuple* tuple, const bool shallow) {
  if (tuple == nullptr) {
    throw std::invalid_argument("The first argument must not be null.");
  }
  if (tuple->aliasCount > 0 || shallow) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* copy = new Tuple;
    copy->refcount = 1;
    copy->aliasCount = 0;
    copy->data = tuple->data;
    return copy;
  }
  tuple->refcount++;
  return tuple;
}

void __quantum__rt__tuple_update_reference_count(Tuple* tuple,
                                                 const int32_t k) {
  if (tuple != nullptr) {
    tuple->refcount += k;
    if (tuple->refcount == 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete tuple;
    }
  }
}

void __quantum__rt__tuple_update_alias_count(Tuple* tuple, const int32_t k) {
  if (tuple != nullptr) {
    tuple->aliasCount += k;
    if (tuple->aliasCount < 0) {
      throw std::invalid_argument("Alias count must be positive: " +
                                  std::to_string(tuple->aliasCount));
    }
  }
}

// *** ARRAYS ***
Array* __quantum__rt__array_create_1d(const int32_t size, const int64_t n) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* array = new Array;
  array->refcount = 1;
  array->aliasCount = 0;
  array->data =
      std::vector(static_cast<size_t>(size * n), static_cast<int8_t>(0));
  array->elementSize = size;
  return array;
}

Array* __quantum__rt__array_copy(Array* array, const bool shallow) {
  if (array == nullptr) {
    throw std::invalid_argument("The first argument must not be null.");
  }
  if (array->aliasCount > 0 || shallow) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* copy = new Array;
    copy->refcount = 1;
    copy->aliasCount = 0;
    copy->data = array->data;
    copy->elementSize = array->elementSize;
    return copy;
  }
  array->refcount++;
  return array;
}

Array* __quantum__rt__array_concatenate(Array* left, Array* right) {
  if (left == nullptr) {
    throw std::invalid_argument("The first argument must not be null.");
  }
  if (right == nullptr) {
    throw std::invalid_argument("The second argument must not be null.");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* array = new Array;
  array->refcount = 1;
  array->aliasCount = 0;
  array->data = std::vector<int8_t>();
  array->data.reserve(left->data.size() + right->data.size());
  std::ranges::copy(std::as_const(left->data), array->data.begin());
  std::ranges::copy(
      std::as_const(right->data),
      array->data.begin() +
          static_cast<std::vector<int8_t>::difference_type>(left->data.size()));
  return array;
}

Array* __quantum__rt__array_slice_1d(Array* array, Range slice,
                                     const bool /*unused*/) {
  if (array == nullptr) {
    throw std::invalid_argument("The first argument must not be null.");
  }
  if (slice.start < 0) {
    throw std::out_of_range("Slice start out of bounds (negative): " +
                            std::to_string(slice.start));
  }
  if (slice.end < 0) {
    throw std::out_of_range("Slice end out of bounds (negative): " +
                            std::to_string(slice.end));
  }
  if (slice.start >= slice.end) {
    throw std::invalid_argument("Slice start must be less than slice end: " +
                                std::to_string(slice.start) +
                                " >= " + std::to_string(slice.end));
  }
  if (slice.start >= __quantum__rt__array_get_size_1d(array)) {
    throw std::out_of_range(
        "Slice start out of bounds (size: " +
        std::to_string(__quantum__rt__array_get_size_1d(array)) +
        "): " + std::to_string(slice.start));
  }
  if (slice.end > __quantum__rt__array_get_size_1d(array)) {
    throw std::out_of_range(
        "Slice end out of bounds (size: " +
        std::to_string(__quantum__rt__array_get_size_1d(array)) +
        "): " + std::to_string(slice.end));
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* sliced = new Array;
  sliced->refcount = 1;
  sliced->aliasCount = 0;
  sliced->elementSize = array->elementSize;
  sliced->data = std::vector<int8_t>(
      array->data.cbegin() + static_cast<std::vector<int8_t>::difference_type>(
                                 array->elementSize * slice.start),
      array->data.cbegin() + static_cast<std::vector<int8_t>::difference_type>(
                                 array->elementSize * slice.end));
  return sliced;
}

int64_t __quantum__rt__array_get_size_1d(const Array* array) {
  return static_cast<int64_t>(array->data.size()) / array->elementSize;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* array, const int64_t i) {
  if (i < 0) {
    throw std::out_of_range("Index out of bounds (negative): " +
                            std::to_string(i));
  }
  if (const auto size = __quantum__rt__array_get_size_1d(array); i >= size) {
    throw std::out_of_range("Index out of bounds (size: " +
                            std::to_string(size) + "): " + std::to_string(i));
  }
  return &array->data[static_cast<size_t>(array->elementSize * i)];
}

void __quantum__rt__array_update_reference_count(Array* array,
                                                 const int32_t k) {
  if (array != nullptr) {
    array->refcount += k;
    if (array->refcount == 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete array;
    }
  }
}

void __quantum__rt__array_update_alias_count(Array* array, const int32_t k) {
  if (array != nullptr) {
    array->aliasCount += k;
    if (array->aliasCount < 0) {
      throw std::invalid_argument("Alias count must be positive: " +
                                  std::to_string(array->aliasCount));
    }
  }
}

Array* __quantum__rt__array_create(int32_t /*unused*/, int32_t /*unused*/,
                                   int64_t* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

int32_t __quantum__rt__array_get_dim(Array* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

int64_t __quantum__rt__array_get_size(Array* /*unused*/, int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

int8_t* __quantum__rt__array_get_element_ptr(Array* /*unused*/,
                                             int64_t* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

Array* __quantum__rt__array_slice(Array* /*unused*/, int32_t /*unused*/,
                                  Range /*unused*/, bool /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

Array* __quantum__rt__array_project(Array* /*unused*/, int32_t /*unused*/,
                                    int64_t /*unused*/, bool /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

// *** CALLABLES ***
Callable*
__quantum__rt__callable_create(void (* /*unused*/[4])(Tuple*, Tuple*, Tuple*),
                               void (* /*unused*/[2])(Tuple*, Tuple*, Tuple*),
                               Tuple* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

Callable* __quantum__rt__callable_copy(Callable* /*unused*/, bool /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__callable_invoke(Callable* /*unused*/, Tuple* /*unused*/,
                                    Tuple* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__callable_make_adjoint(Callable* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__callable_make_controlled(Callable* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__callable_update_reference_count(Callable* /*unused*/,
                                                    const int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__callable_update_alias_count(Callable* /*unused*/,
                                                const int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__capture_update_reference_count(Callable* /*unused*/,
                                                   const int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__capture_update_alias_count(Callable* /*unused*/,
                                               const int32_t /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

// *** CLASSICAL RUNTIME ***
void __quantum__rt__message(const String* msg) { std::cout << *msg << "\n"; }

void __quantum__rt__fail(const String* msg) {
  throw std::runtime_error(msg->content);
}

// *** QUANTUM INSTRUCTION SET AND RUNTIME ***
Qubit* __quantum__rt__qubit_allocate() {
  auto& backend = qir::Backend::getInstance();
  return backend.qAlloc();
}

Array* __quantum__rt__qubit_allocate_array(const int64_t n) {
  auto* array = __quantum__rt__array_create_1d(sizeof(Qubit*), n);
  for (int64_t i = 0; i < n; ++i) {
    auto* const q = reinterpret_cast<Qubit**>(
        __quantum__rt__array_get_element_ptr_1d(array, i));
    *q = __quantum__rt__qubit_allocate();
  }
  return array;
}

void __quantum__rt__qubit_release(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.qFree(qubit);
}

void __quantum__rt__qubit_release_array(Array* array) {
  const auto size = __quantum__rt__array_get_size_1d(array);
  // deallocate every qubit
  for (int64_t i = 0; i < size; ++i) {
    auto* const q = reinterpret_cast<Qubit**>(
        __quantum__rt__array_get_element_ptr_1d(array, i));
    __quantum__rt__qubit_release(*q);
  }
  // deallocate array
  __quantum__rt__array_update_reference_count(array, -1);
}

// QUANTUM INSTRUCTION SET
void __quantum__qis__x__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::X, qubit);
}

void __quantum__qis__y__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Y, qubit);
}

void __quantum__qis__z__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Z, qubit);
}

void __quantum__qis__h__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::H, qubit);
}

void __quantum__qis__s__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::S, qubit);
}

void __quantum__qis__sdg__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Sdg, qubit);
}

void __quantum__qis__sx__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::SX, qubit);
}

void __quantum__qis__sxdg__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::SXdg, qubit);
}

void __quantum__qis__sqrtx__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::SX, qubit);
}

void __quantum__qis__sqrtxdg__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::SXdg, qubit);
}

void __quantum__qis__t__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::T, qubit);
}

void __quantum__qis__tdg__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Tdg, qubit);
}

void __quantum__qis__rx__body(Qubit* qubit, const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RX, phi, qubit);
}

void __quantum__qis__ry__body(Qubit* qubit, const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RY, phi, qubit);
}

void __quantum__qis__rz__body(Qubit* qubit, const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RZ, phi, qubit);
}

void __quantum__qis__p__body(Qubit* qubit, const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::P, phi, qubit);
}

void __quantum__qis__rxx__body(Qubit* target1, Qubit* target2,
                               const double theta) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RXX, theta, target1, target2);
}

void __quantum__qis__u__body(Qubit* qubit, const double theta, const double phi,
                             const double lambda) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::U, theta, phi, lambda, qubit);
}

void __quantum__qis__u3__body(Qubit* qubit, const double theta,
                              const double phi, const double lambda) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::U, theta, phi, lambda, qubit);
}

void __quantum__qis__u2__body(Qubit* qubit, const double theta,
                              const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::U2, theta, phi, qubit);
}

void __quantum__qis__u1__body(Qubit* qubit, const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::P, phi, qubit);
}

void __quantum__qis__cu1__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::P, phi, control, target);
}

void __quantum__qis__cu3__body(Qubit* control, Qubit* target,
                               const double theta, const double phi,
                               const double lambda) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::U, theta, phi, lambda, control, target);
}

void __quantum__qis__cnot__body(Qubit* control, Qubit* target) {
  __quantum__qis__cx__body(control, target);
}

void __quantum__qis__cx__body(Qubit* control, Qubit* target) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::X, control, target);
}

void __quantum__qis__cy__body(Qubit* control, Qubit* target) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Y, control, target);
}

void __quantum__qis__cz__body(Qubit* control, Qubit* target) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Z, control, target);
}

void __quantum__qis__ch__body(Qubit* control, Qubit* target) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::H, control, target);
}

void __quantum__qis__swap__body(Qubit* target1, Qubit* target2) {
  auto& backend = qir::Backend::getInstance();
  backend.swap(target1, target2);
}

void __quantum__qis__cswap__body(Qubit* control, Qubit* target1,
                                 Qubit* target2) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::SWAP, control, target1, target2);
}

void __quantum__qis__crz__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RZ, phi, control, target);
}

void __quantum__qis__cry__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RY, phi, control, target);
}

void __quantum__qis__crx__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RX, phi, control, target);
}

void __quantum__qis__cp__body(Qubit* control, Qubit* target, const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::P, phi, control, target);
}

void __quantum__qis__rzz__body(Qubit* target1, Qubit* target2,
                               const double phi) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::RZZ, phi, target1, target2);
}

void __quantum__qis__ccx__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::X, control1, control2, target);
}

void __quantum__qis__ccz__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& backend = qir::Backend::getInstance();
  backend.apply(qc::Z, control1, control2, target);
}

void __quantum__qis__mz__body(Qubit* qubit, Result* result) {
  auto& backend = qir::Backend::getInstance();
  backend.measure(qubit, result);
}

Result* __quantum__qis__m__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  auto* result = backend.rAlloc();
  __quantum__qis__mz__body(qubit, result);
  return result;
}

Result* __quantum__qis__measure__body(Qubit* qubit) {
  return __quantum__qis__m__body(qubit);
}

void __quantum__qis__reset__body(Qubit* qubit) {
  auto& backend = qir::Backend::getInstance();
  backend.reset<1>({qubit});
}

void __quantum__rt__initialize(char* /*unused*/) {
  qir::Backend::getInstance().resetBackend();
}

bool __quantum__rt__read_result(Result* result) {
  auto& backend = qir::Backend::getInstance();
  return backend.deref(result).r;
}

void __quantum__rt__tuple_record_output(int64_t /*unused*/,
                                        const char* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__array_record_output(int64_t /*unused*/,
                                        const char* /*unused*/) {
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__
     << " not implemented.";
  __quantum__rt__fail(__quantum__rt__string_create(ss.str().c_str()));
}

void __quantum__rt__result_record_output(Result* result, const char* label) {
  std::cout << label << ": " << (__quantum__rt__read_result(result) ? 1 : 0)
            << "\n";
}

void __quantum__rt__bool_record_output(bool value, const char* label) {
  std::cout << label << ": " << (value ? "true" : "false") << "\n";
}

} // extern "C"
