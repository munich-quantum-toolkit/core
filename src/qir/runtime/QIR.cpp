/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/runtime/QIR.h"

#include "ir/operations/OpType.hpp"
#include "qir/runtime/Runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

// *** MEASUREMENT RESULTS ***
Result* __quantum__rt__result_get_zero() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Result*>(qir::Runtime::RESULT_ZERO_ADDRESS);
}

Result* __quantum__rt__result_get_one() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Result*>(qir::Runtime::RESULT_ONE_ADDRESS);
}

bool __quantum__rt__result_equal(Result* result1, Result* result2) {
  auto& runtime = qir::Runtime::getInstance();
  return runtime.equal(result1, result2);
}

void __quantum__rt__result_update_reference_count(Result* result,
                                                  const int32_t k) {
  auto& runtime = qir::Runtime::getInstance();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  if (result != nullptr &&
      result != reinterpret_cast<Result*>(qir::Runtime::RESULT_ZERO_ADDRESS) &&
      result != reinterpret_cast<Result*>(qir::Runtime::RESULT_ONE_ADDRESS)) {
    // NOLINTEND(performance-no-int-to-ptr)
    auto& refcount = runtime.deref(result).refcount;
    refcount += k;
    if (refcount == 0) {
      runtime.rFree(result);
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

// *** QUANTUM INSTRUCTION SET AND RUNTIME ***
Qubit* __quantum__rt__qubit_allocate() {
  auto& runtime = qir::Runtime::getInstance();
  return runtime.qAlloc();
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
  auto& runtime = qir::Runtime::getInstance();
  runtime.qFree(qubit);
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
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::X>(qubit);
}

void __quantum__qis__y__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Y>(qubit);
}

void __quantum__qis__z__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Z>(qubit);
}

void __quantum__qis__h__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::H>(qubit);
}

void __quantum__qis__s__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::S>(qubit);
}

void __quantum__qis__sdg__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Sdg>(qubit);
}

void __quantum__qis__sx__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::SX>(qubit);
}

void __quantum__qis__sxdg__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::SXdg>(qubit);
}

void __quantum__qis__sqrtx__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::SX>(qubit);
}

void __quantum__qis__sqrtxdg__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::SXdg>(qubit);
}

void __quantum__qis__t__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::T>(qubit);
}

void __quantum__qis__tdg__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Tdg>(qubit);
}

void __quantum__qis__rx__body(Qubit* qubit, const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RX>(phi, qubit);
}

void __quantum__qis__ry__body(Qubit* qubit, const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RY>(phi, qubit);
}

void __quantum__qis__rz__body(Qubit* qubit, const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RZ>(phi, qubit);
}

void __quantum__qis__p__body(Qubit* qubit, const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::P>(phi, qubit);
}

void __quantum__qis__rxx__body(Qubit* target1, Qubit* target2,
                               const double theta) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RXX>(theta, target1, target2);
}

void __quantum__qis__ryy__body(Qubit* target1, Qubit* target2,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RYY>(phi, target1, target2);
}

void __quantum__qis__rzz__body(Qubit* target1, Qubit* target2,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RZZ>(phi, target1, target2);
}

void __quantum__qis__rzx__body(Qubit* target1, Qubit* target2,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RZX>(phi, target1, target2);
}

void __quantum__qis__u__body(Qubit* qubit, const double theta, const double phi,
                             const double lambda) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::U>(theta, phi, lambda, qubit);
}

void __quantum__qis__u3__body(Qubit* qubit, const double theta,
                              const double phi, const double lambda) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::U>(theta, phi, lambda, qubit);
}

void __quantum__qis__u2__body(Qubit* qubit, const double theta,
                              const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::U2>(theta, phi, qubit);
}

void __quantum__qis__u1__body(Qubit* qubit, const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::P>(phi, qubit);
}

void __quantum__qis__cu1__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::P>(phi, control, target);
}

void __quantum__qis__cu3__body(Qubit* control, Qubit* target,
                               const double theta, const double phi,
                               const double lambda) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::U>(theta, phi, lambda, control, target);
}

void __quantum__qis__cnot__body(Qubit* control, Qubit* target) {
  __quantum__qis__cx__body(control, target);
}

void __quantum__qis__cx__body(Qubit* control, Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::X>(control, target);
}

void __quantum__qis__cy__body(Qubit* control, Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Y>(control, target);
}

void __quantum__qis__cz__body(Qubit* control, Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Z>(control, target);
}

void __quantum__qis__ch__body(Qubit* control, Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::H>(control, target);
}

void __quantum__qis__swap__body(Qubit* target1, Qubit* target2) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.swap(target1, target2);
}

void __quantum__qis__cswap__body(Qubit* control, Qubit* target1,
                                 Qubit* target2) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::SWAP>(control, target1, target2);
}

void __quantum__qis__crz__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RZ>(phi, control, target);
}

void __quantum__qis__cry__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RY>(phi, control, target);
}

void __quantum__qis__crx__body(Qubit* control, Qubit* target,
                               const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::RX>(phi, control, target);
}

void __quantum__qis__cp__body(Qubit* control, Qubit* target, const double phi) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::P>(phi, control, target);
}

void __quantum__qis__ccx__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::X>(control1, control2, target);
}

void __quantum__qis__ccy__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Y>(control1, control2, target);
}

void __quantum__qis__ccz__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.apply<qc::Z>(control1, control2, target);
}

void __quantum__qis__mz__body(Qubit* qubit, Result* result) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.measure(qubit, result);
}

Result* __quantum__qis__m__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  auto* result = runtime.rAlloc();
  __quantum__qis__mz__body(qubit, result);
  return result;
}

Result* __quantum__qis__measure__body(Qubit* qubit) {
  return __quantum__qis__m__body(qubit);
}

void __quantum__qis__reset__body(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.reset<1>({qubit});
}

void __quantum__rt__initialize(char* /*unused*/) {
  qir::Runtime::getInstance().reset();
}

bool __quantum__rt__read_result(Result* result) {
  auto& runtime = qir::Runtime::getInstance();
  return runtime.deref(result).r;
}

void __quantum__rt__result_record_output(Result* result, const char* label) {
  std::cout << label << ": " << (__quantum__rt__read_result(result) ? 1 : 0)
            << "\n";
}

} // extern "C"
