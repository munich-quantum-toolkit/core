/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Execution/Runtime/QIR.h"

#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/QIR/Execution/Runtime/Runtime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct alignas(std::max_align_t) TupleHeader {
  int32_t referenceCount = 1;
  int64_t size = 0;
};

} // namespace

static auto getTupleHeader(Tuple* tuple) -> TupleHeader* {
  return reinterpret_cast<TupleHeader*>(tuple) - 1;
}

static auto controlsFromArray(Array* array) -> std::vector<Qubit*> {
  if (array == nullptr) {
    throw std::invalid_argument("QIR control array must not be null");
  }
  if (std::cmp_not_equal(array->elementSize, sizeof(Qubit*))) {
    throw std::invalid_argument(
        "QIR control array elements must contain qubit pointers");
  }
  const auto size = __quantum__rt__array_get_size_1d(array);
  std::vector<Qubit*> controls(static_cast<std::size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    const auto* element = __quantum__rt__array_get_element_ptr_1d(array, i);
    if (element == nullptr) {
      throw std::out_of_range("QIR control array index out of range");
    }
    std::memcpy(static_cast<void*>(&controls[static_cast<std::size_t>(i)]),
                element, sizeof(Qubit*));
  }
  return controls;
}

static auto applyControlled(const qc::OpType op, Array* controlArray,
                            Qubit* target, const std::span<const qc::fp> params)
    -> void {
  const auto controls = controlsFromArray(controlArray);
  const std::array targets{target};
  qir::Runtime::getInstance().apply(op, params, controls, targets);
}

template <std::size_t NumParams, std::size_t NumTargets>
static auto applyControlledTuple(const qc::OpType op, Array* controls,
                                 Tuple* tuple) -> void {
  if (tuple == nullptr) {
    throw std::invalid_argument(
        "QIR generic controlled argument tuple must not be null");
  }
  const auto validateSize = [&](const std::size_t expected) {
    if (std::cmp_not_equal(getTupleHeader(tuple)->size, expected)) {
      throw std::invalid_argument(
          "QIR generic controlled argument tuple has an invalid size");
    }
  };

  if constexpr (NumParams == 0) {
    struct Args {
      std::array<Qubit*, NumTargets> targets{};
    };
    static_assert(std::is_standard_layout_v<Args>);
    validateSize(sizeof(Args));
    Args args;
    std::memcpy(&args, tuple, sizeof(Args));
    const auto controlList = controlsFromArray(controls);
    qir::Runtime::getInstance().apply(op, {}, controlList, args.targets);
  } else {
    struct Args {
      std::array<qc::fp, NumParams> parameters{};
      std::array<Qubit*, NumTargets> targets{};
    };
    static_assert(std::is_standard_layout_v<Args>);
    validateSize(sizeof(Args));
    Args args;
    std::memcpy(&args, tuple, sizeof(Args));
    const auto controlList = controlsFromArray(controls);
    qir::Runtime::getInstance().apply(op, args.parameters, controlList,
                                      args.targets);
  }
}

extern "C" {

// *** ARRAYS ***
Array* __quantum__rt__array_create_1d(const int32_t size, const int64_t n) {
  if (size <= 0 || n < 0) {
    throw std::invalid_argument(
        "QIR array element size must be positive and length nonnegative");
  }
  const auto elementSize = static_cast<std::size_t>(size);
  const auto length = static_cast<std::size_t>(n);
  constexpr auto maxObjectSize =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (length > maxObjectSize / elementSize) {
    throw std::length_error("QIR array allocation size overflow");
  }
  auto* array = new Array;
  array->refcount = 1;
  array->data = std::vector(length * elementSize, static_cast<int8_t>(0));
  array->elementSize = size;
  return array;
}

int64_t __quantum__rt__array_get_size_1d(const Array* array) {
  return static_cast<int64_t>(array->data.size()) / array->elementSize;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* array, const int64_t i) {
  if (array == nullptr || i < 0 ||
      i >= __quantum__rt__array_get_size_1d(array)) {
    return nullptr;
  }
  return &array->data[static_cast<size_t>(array->elementSize * i)];
}

void __quantum__rt__array_update_reference_count(Array* array,
                                                 const int32_t k) {
  if (array != nullptr) {
    array->refcount += k;
    if (array->refcount == 0) {
      delete array;
    }
  }
}

Tuple* __quantum__rt__tuple_create(const int64_t size) {
  if (size < 0) {
    throw std::invalid_argument("QIR tuple size must not be negative");
  }
  const auto payloadSize = static_cast<std::size_t>(size);
  constexpr auto maxObjectSize =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (payloadSize > maxObjectSize - sizeof(TupleHeader)) {
    throw std::length_error("QIR tuple allocation size overflow");
  }
  const auto bytes = sizeof(TupleHeader) + payloadSize;
  auto* storage = static_cast<std::byte*>(
      ::operator new(bytes, std::align_val_t{alignof(TupleHeader)}));
  auto* header = std::construct_at(reinterpret_cast<TupleHeader*>(storage));
  header->size = size;
  auto* payload =
      std::next(storage, static_cast<std::ptrdiff_t>(sizeof(TupleHeader)));
  std::ranges::fill_n(payload, size, std::byte{0});
  return reinterpret_cast<Tuple*>(payload);
}

void __quantum__rt__tuple_update_reference_count(Tuple* tuple,
                                                 const int32_t k) {
  if (tuple == nullptr) {
    return;
  }
  auto* header = getTupleHeader(tuple);
  header->referenceCount += k;
  if (header->referenceCount == 0) {
    std::destroy_at(header);
    ::operator delete(header, std::align_val_t{alignof(TupleHeader)});
  }
}

// *** QUANTUM INSTRUCTION SET AND RUNTIME ***
Qubit* __quantum__rt__qubit_allocate(bool* outError) {
  if (outError != nullptr) {
    *outError = false;
  }
  auto& runtime = qir::Runtime::getInstance();
  return runtime.qAlloc();
}

void __quantum__rt__qubit_array_allocate(const int64_t size, Qubit** array,
                                         bool* outError) {
  if (outError != nullptr) {
    *outError = false;
  }
  if (size < 0 || (size > 0 && array == nullptr)) {
    if (outError != nullptr) {
      *outError = true;
      return;
    }
    throw std::invalid_argument("Invalid QIR qubit array allocation");
  }
  for (auto*& qubit : std::span(array, static_cast<std::size_t>(size))) {
    qubit = qir::Runtime::getInstance().qAlloc();
  }
}

void __quantum__rt__qubit_array_release(const int64_t size, Qubit** array) {
  if (size < 0 || (size > 0 && array == nullptr)) {
    throw std::invalid_argument("Invalid QIR qubit array release");
  }
  for (Qubit* qubit : std::span(array, static_cast<std::size_t>(size))) {
    qir::Runtime::getInstance().qFree(qubit);
  }
}

Result* __quantum__rt__result_allocate(bool* outError) {
  if (outError != nullptr) {
    *outError = false;
  }
  return qir::Runtime::getInstance().rAlloc();
}

void __quantum__rt__result_release(Result* result) {
  qir::Runtime::getInstance().rFree(result);
}

void __quantum__rt__result_array_allocate(const int64_t size, Result** array,
                                          bool* outError) {
  if (outError != nullptr) {
    *outError = false;
  }
  if (size < 0 || (size > 0 && array == nullptr)) {
    if (outError != nullptr) {
      *outError = true;
      return;
    }
    throw std::invalid_argument("Invalid QIR result array allocation");
  }
  for (auto*& result : std::span(array, static_cast<std::size_t>(size))) {
    result = qir::Runtime::getInstance().rAlloc();
  }
}

void __quantum__rt__result_array_release(const int64_t size, Result** array) {
  if (size < 0 || (size > 0 && array == nullptr)) {
    throw std::invalid_argument("Invalid QIR result array release");
  }
  for (Result* result : std::span(array, static_cast<std::size_t>(size))) {
    qir::Runtime::getInstance().rFree(result);
  }
}

void __quantum__rt__qubit_release(Qubit* qubit) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.qFree(qubit);
}

// QUANTUM INSTRUCTION SET
#define MQT_QIR_DEFINE_1_0(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit* target) {                     \
    qir::Runtime::getInstance().apply<qc::OP>(target);                         \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit* control, Qubit* target) {    \
    qir::Runtime::getInstance().apply<qc::OP>(control, target);                \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit* control0, Qubit* control1,  \
                                            Qubit* target) {                   \
    qir::Runtime::getInstance().apply<qc::OP>(control0, control1, target);     \
  }
#define MQT_QIR_DEFINE_1_1(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, Qubit* target) {          \
    qir::Runtime::getInstance().apply<qc::OP>(p0, target);                     \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, Qubit* control,          \
                                           Qubit* target) {                    \
    qir::Runtime::getInstance().apply<qc::OP>(p0, control, target);            \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, Qubit* control0,        \
                                            Qubit* control1, Qubit* target) {  \
    qir::Runtime::getInstance().apply<qc::OP>(p0, control0, control1, target); \
  }
#define MQT_QIR_DEFINE_1_2(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, double p1,                \
                                          Qubit* target) {                     \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, target);                 \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, double p1,               \
                                           Qubit* control, Qubit* target) {    \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, control, target);        \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(                                   \
      double p0, double p1, Qubit* control0, Qubit* control1, Qubit* target) { \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, control0, control1,      \
                                              target);                         \
  }
#define MQT_QIR_DEFINE_1_3(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, double p1, double p2,     \
                                          Qubit* target) {                     \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, p2, target);             \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, double p1, double p2,    \
                                           Qubit* control, Qubit* target) {    \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, p2, control, target);    \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, double p1, double p2,   \
                                            Qubit* control0, Qubit* control1,  \
                                            Qubit* target) {                   \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, p2, control0, control1,  \
                                              target);                         \
  }
#define MQT_QIR_DEFINE_2_0(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit* target0, Qubit* target1) {    \
    qir::Runtime::getInstance().apply<qc::OP>(target0, target1);               \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit* control, Qubit* target0,     \
                                           Qubit* target1) {                   \
    qir::Runtime::getInstance().apply<qc::OP>(control, target0, target1);      \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit* control0, Qubit* control1,  \
                                            Qubit* target0, Qubit* target1) {  \
    qir::Runtime::getInstance().apply<qc::OP>(control0, control1, target0,     \
                                              target1);                        \
  }
#define MQT_QIR_DEFINE_2_1(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, Qubit* target0,           \
                                          Qubit* target1) {                    \
    qir::Runtime::getInstance().apply<qc::OP>(p0, target0, target1);           \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, Qubit* control,          \
                                           Qubit* target0, Qubit* target1) {   \
    qir::Runtime::getInstance().apply<qc::OP>(p0, control, target0, target1);  \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, Qubit* control0,        \
                                            Qubit* control1, Qubit* target0,   \
                                            Qubit* target1) {                  \
    qir::Runtime::getInstance().apply<qc::OP>(p0, control0, control1, target0, \
                                              target1);                        \
  }
#define MQT_QIR_DEFINE_2_2(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, double p1,                \
                                          Qubit* target0, Qubit* target1) {    \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, target0, target1);       \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(                                    \
      double p0, double p1, Qubit* control, Qubit* target0, Qubit* target1) {  \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, control, target0,        \
                                              target1);                        \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, double p1,              \
                                            Qubit* control0, Qubit* control1,  \
                                            Qubit* target0, Qubit* target1) {  \
    qir::Runtime::getInstance().apply<qc::OP>(p0, p1, control0, control1,      \
                                              target0, target1);               \
  }
#define MQT_QIR_DEFINE_3_0(NAME, OP, SUFFIX)                                   \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit* target0, Qubit* target1,      \
                                          Qubit* target2) {                    \
    qir::Runtime::getInstance().apply<qc::OP>(target0, target1, target2);      \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit* control, Qubit* target0,     \
                                           Qubit* target1, Qubit* target2) {   \
    qir::Runtime::getInstance().apply<qc::OP>(control, target0, target1,       \
                                              target2);                        \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit* control0, Qubit* control1,  \
                                            Qubit* target0, Qubit* target1,    \
                                            Qubit* target2) {                  \
    qir::Runtime::getInstance().apply<qc::OP>(control0, control1, target0,     \
                                              target1, target2);               \
  }
#define MQT_QIR_DEFINE_CTL_1_0(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Qubit* target) {                 \
    applyControlled(qc::OP, controls, target, {});                             \
  }
#define MQT_QIR_DEFINE_CTL_1_1(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<1, 1>(qc::OP, controls, args);                        \
  }
#define MQT_QIR_DEFINE_CTL_1_2(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<2, 1>(qc::OP, controls, args);                        \
  }
#define MQT_QIR_DEFINE_CTL_1_3(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<3, 1>(qc::OP, controls, args);                        \
  }
#define MQT_QIR_DEFINE_CTL_2_0(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<0, 2>(qc::OP, controls, args);                        \
  }
#define MQT_QIR_DEFINE_CTL_2_1(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<1, 2>(qc::OP, controls, args);                        \
  }
#define MQT_QIR_DEFINE_CTL_2_2(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<2, 2>(qc::OP, controls, args);                        \
  }
#define MQT_QIR_DEFINE_CTL_3_0(NAME, OP, CTL_SUFFIX)                           \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls, Tuple* args) {  \
    applyControlledTuple<0, 3>(qc::OP, controls, args);                        \
  }

#define MQT_GATE(KEY, NAME, OP, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)   \
  MQT_QIR_DEFINE_##TARGETS##_##PARAMS(NAME, OP, SUFFIX)                        \
      MQT_QIR_DEFINE_CTL_##TARGETS##_##PARAMS(NAME, OP, CTL_SUFFIX)
#include "mlir/Conversion/GateTable.def"

#undef MQT_QIR_DEFINE_1_0
#undef MQT_QIR_DEFINE_1_1
#undef MQT_QIR_DEFINE_1_2
#undef MQT_QIR_DEFINE_1_3
#undef MQT_QIR_DEFINE_2_0
#undef MQT_QIR_DEFINE_2_1
#undef MQT_QIR_DEFINE_2_2
#undef MQT_QIR_DEFINE_3_0
#undef MQT_QIR_DEFINE_CTL_1_0
#undef MQT_QIR_DEFINE_CTL_1_1
#undef MQT_QIR_DEFINE_CTL_1_2
#undef MQT_QIR_DEFINE_CTL_1_3
#undef MQT_QIR_DEFINE_CTL_2_0
#undef MQT_QIR_DEFINE_CTL_2_1
#undef MQT_QIR_DEFINE_CTL_2_2
#undef MQT_QIR_DEFINE_CTL_3_0

void __quantum__qis__gphase__body(const double phase) {
  qir::Runtime::getInstance().applyGlobalPhase(phase);
}

void __quantum__qis__cnot__body(Qubit* control, Qubit* target) {
  __quantum__qis__cx__body(control, target);
}

void __quantum__qis__mz__body(Qubit* qubit, Result* result) {
  auto& runtime = qir::Runtime::getInstance();
  runtime.measure(qubit, result);
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
  const bool bit = __quantum__rt__read_result(result);
  auto& runtime = qir::Runtime::getInstance();
  runtime.outputResult(bit, label);
  // Accumulate new measurement bit.
  runtime.appendMeasurementBit(bit);
}

void __quantum__rt__bool_record_output(bool value, const char* label) {
  qir::Runtime::getInstance().outputBool(value, label);
}

void __quantum__rt__int_record_output(int64_t value, const char* label) {
  qir::Runtime::getInstance().outputInt(value, label);
}

void __quantum__rt__double_record_output(double value, const char* label) {
  qir::Runtime::getInstance().outputFloat(value, label);
}

void __quantum__rt__tuple_record_output(int64_t elementCount,
                                        const char* label) {
  qir::Runtime::getInstance().outputTuple(elementCount, label);
}

void __quantum__rt__array_record_output(int64_t size, const char* label) {
  qir::Runtime::getInstance().outputArray(size, label);
}

void __quantum__rt__result_array_record_output(const int64_t size,
                                               Result** results,
                                               const char* label) {
  if (size < 0 || (size > 0 && results == nullptr)) {
    throw std::invalid_argument("Invalid QIR result array output");
  }
  auto& runtime = qir::Runtime::getInstance();
  std::string values;
  values.reserve(static_cast<std::size_t>(size));
  for (Result* result : std::span(results, static_cast<std::size_t>(size))) {
    const auto value = runtime.deref(result).r;
    values.push_back(value ? '1' : '0');
    runtime.appendMeasurementBit(value);
  }
  runtime.outputResultArray(values, label);
}

} // extern "C"
