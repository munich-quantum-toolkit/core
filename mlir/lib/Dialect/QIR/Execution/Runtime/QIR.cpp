/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/Execution/Runtime/QIR.h"

#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/DDAdapter.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

[[noreturn]] static void fail(const char* message) noexcept {
  std::fputs("QIR runtime failure: ", stderr);
  std::fputs(message, stderr);
  std::fputc('\n', stderr);
  std::abort();
}

// Keep C++ exceptions inside the runtime, never across generated QIR frames.
template <typename Function>
static decltype(auto) runtimeCall(Function&& operation) noexcept {
  try {
    return std::forward<Function>(operation)();
  } catch (const std::exception& error) {
    fail(error.what());
  } catch (...) {
    fail("Unknown runtime failure");
  }
}

static auto getTupleHeader(Tuple* tuple) -> TupleHeader* {
  return reinterpret_cast<TupleHeader*>(tuple) - 1;
}

static auto controlsFromArray(Array* array) -> llvm::SmallVector<Qubit*> {
  if (array == nullptr) {
    fail("QIR control array must not be null");
  }
  if (std::cmp_not_equal(array->elementSize, sizeof(Qubit*))) {
    fail("QIR control array elements must contain qubit pointers");
  }
  const auto size = __quantum__rt__array_get_size_1d(array);
  llvm::SmallVector<Qubit*> controls(static_cast<std::size_t>(size));
  if (!controls.empty()) {
    std::memcpy(static_cast<void*>(controls.data()), array->data.data(),
                array->data.size());
  }
  return controls;
}

template <typename GateOp>
static auto applyGateMatrix(llvm::ArrayRef<double> parameters,
                            std::span<Qubit* const> controls,
                            std::span<Qubit* const> targets) -> void {
  runtimeCall([&] {
    auto& runtime = qir::Runtime::getInstance();
    if constexpr (std::is_same_v<GateOp, mlir::qco::SWAPOp>) {
      if (controls.empty() && targets.size() == 2) {
        runtime.swap(targets[0], targets[1]);
        return;
      }
    }
    runtime.apply(mlir::qco::getStandardGateMatrix<GateOp>(parameters),
                  controls, targets);
  });
}

template <typename GateOp, size_t NumTargets, typename... Args>
static auto applyGate(Args... args) -> void {
  auto parameters = qir::packOfType<double>(args...);
  auto qubits = qir::packOfType<Qubit*>(args...);
  static_assert(parameters.size() + qubits.size() == sizeof...(Args),
                "Parameters must precede the gate's qubits");
  static_assert(qubits.size() >= NumTargets,
                "Not enough qubits provided for the gate");
  const auto numControls = qubits.size() - NumTargets;
  applyGateMatrix<GateOp>(
      parameters, std::span<Qubit* const>{qubits.data(), numControls},
      std::span<Qubit* const>{qubits.data() + numControls, NumTargets});
}

template <typename GateOp>
static auto applyControlled(Array* controlArray, Qubit* target) -> void {
  const auto controls = controlsFromArray(controlArray);
  const std::array targets{target};
  applyGateMatrix<GateOp>({}, controls, targets);
}

template <typename GateOp, size_t NumParams, size_t NumTargets>
static auto applyControlledTuple(Array* controls, Tuple* tuple) -> void {
  if (tuple == nullptr) {
    fail("QIR generic controlled argument tuple must not be null");
  }
  const auto validateSize = [&](const std::size_t expected) {
    if (std::cmp_not_equal(getTupleHeader(tuple)->size, expected)) {
      fail("QIR generic controlled argument tuple has an invalid size");
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
    applyGateMatrix<GateOp>({}, controlList, args.targets);
  } else {
    struct Args {
      std::array<double, NumParams> parameters{};
      std::array<Qubit*, NumTargets> targets{};
    };
    static_assert(std::is_standard_layout_v<Args>);
    validateSize(sizeof(Args));
    Args args;
    std::memcpy(&args, tuple, sizeof(Args));
    const auto controlList = controlsFromArray(controls);
    applyGateMatrix<GateOp>(args.parameters, controlList, args.targets);
  }
}

// Only explicit resource errors leave the runtime reusable.
template <typename Resource>
static Resource* allocate(bool* outError) noexcept {
  if (outError != nullptr) {
    *outError = false;
  }
  try {
    auto& runtime = qir::Runtime::getInstance();
    if constexpr (std::is_same_v<Resource, Qubit>) {
      return runtime.qAlloc();
    } else {
      return runtime.rAlloc();
    }
  } catch (const std::logic_error& error) {
    if (outError == nullptr) {
      fail(error.what());
    }
    *outError = true;
    return nullptr;
  }
}

template <typename Resource>
static void releaseArray(int64_t size, Resource** array) noexcept {
  if (size < 0 || (size > 0 && array == nullptr)) {
    fail("Invalid QIR resource array release");
  }
  runtimeCall([&] {
    auto& runtime = qir::Runtime::getInstance();
    for (auto* resource : std::span(array, static_cast<size_t>(size))) {
      if constexpr (std::is_same_v<Resource, Qubit>) {
        runtime.qFree(resource);
      } else {
        runtime.rFree(resource);
      }
    }
  });
}

template <typename Resource>
static void allocateArray(int64_t size, Resource** array,
                          bool* outError) noexcept {
  if (outError != nullptr) {
    *outError = false;
  }
  if (size < 0 || (size > 0 && array == nullptr)) {
    if (outError == nullptr) {
      fail("Invalid QIR resource array allocation");
    }
    *outError = true;
    return;
  }
  for (int64_t i = 0; i < size; ++i) {
    array[i] = allocate<Resource>(outError);
    if (outError != nullptr && *outError) {
      releaseArray(i, array);
      std::fill_n(array, i, nullptr);
      return;
    }
  }
}

extern "C" {

// *** ARRAYS ***
Array* __quantum__rt__array_create_1d(const int32_t size,
                                      const int64_t n) noexcept {
  if (size <= 0 || n < 0) {
    fail("QIR array element size must be positive and length nonnegative");
  }
  const auto elementSize = static_cast<std::size_t>(size);
  const auto length = static_cast<std::size_t>(n);
  constexpr auto maxObjectSize =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (length > maxObjectSize / elementSize) {
    fail("QIR array allocation size overflow");
  }
  auto array = std::make_unique<Array>();
  array->refcount = 1;
  array->data = std::vector(length * elementSize, static_cast<int8_t>(0));
  array->elementSize = size;
  return array.release();
}

int64_t __quantum__rt__array_get_size_1d(const Array* array) noexcept {
  return static_cast<int64_t>(array->data.size()) / array->elementSize;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* array,
                                                const int64_t i) noexcept {
  if (array == nullptr || i < 0 ||
      i >= __quantum__rt__array_get_size_1d(array)) {
    return nullptr;
  }
  return &array->data[static_cast<size_t>(array->elementSize * i)];
}

void __quantum__rt__array_update_reference_count(Array* array,
                                                 const int32_t k) noexcept {
  if (array != nullptr) {
    array->refcount += k;
    if (array->refcount == 0) {
      delete array;
    }
  }
}

Tuple* __quantum__rt__tuple_create(const int64_t size) noexcept {
  if (size < 0) {
    fail("QIR tuple size must not be negative");
  }
  const auto payloadSize = static_cast<std::size_t>(size);
  constexpr auto maxObjectSize =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (payloadSize > maxObjectSize - sizeof(TupleHeader)) {
    fail("QIR tuple allocation size overflow");
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
                                                 const int32_t k) noexcept {
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
Qubit* __quantum__rt__qubit_allocate(bool* outError) noexcept {
  return allocate<Qubit>(outError);
}

void __quantum__rt__qubit_array_allocate(const int64_t size, Qubit** array,
                                         bool* outError) noexcept {
  allocateArray(size, array, outError);
}

void __quantum__rt__qubit_array_release(const int64_t size,
                                        Qubit** array) noexcept {
  releaseArray(size, array);
}

Result* __quantum__rt__result_allocate(bool* outError) noexcept {
  return allocate<Result>(outError);
}

void __quantum__rt__result_release(Result* result) noexcept {
  return runtimeCall([&] { qir::Runtime::getInstance().rFree(result); });
}

void __quantum__rt__result_array_allocate(const int64_t size, Result** array,
                                          bool* outError) noexcept {
  allocateArray(size, array, outError);
}

void __quantum__rt__result_array_release(const int64_t size,
                                         Result** array) noexcept {
  releaseArray(size, array);
}

void __quantum__rt__qubit_release(Qubit* qubit) noexcept {
  return runtimeCall([&] {
    auto& runtime = qir::Runtime::getInstance();
    runtime.qFree(qubit);
  });
}

// QUANTUM INSTRUCTION SET
#define MQT_QIR_DEFINE_1_0(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit* target) noexcept {            \
    applyGate<mlir::qco::KEY##Op, 1>(target);                                  \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit* control,                     \
                                           Qubit* target) noexcept {           \
    applyGate<mlir::qco::KEY##Op, 1>(control, target);                         \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit* control0, Qubit* control1,  \
                                            Qubit* target) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 1>(control0, control1, target);              \
  }
#define MQT_QIR_DEFINE_1_1(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, Qubit* target) noexcept { \
    applyGate<mlir::qco::KEY##Op, 1>(p0, target);                              \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, Qubit* control,          \
                                           Qubit* target) noexcept {           \
    applyGate<mlir::qco::KEY##Op, 1>(p0, control, target);                     \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(                                   \
      double p0, Qubit* control0, Qubit* control1, Qubit* target) noexcept {   \
    applyGate<mlir::qco::KEY##Op, 1>(p0, control0, control1, target);          \
  }
#define MQT_QIR_DEFINE_1_2(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, double p1,                \
                                          Qubit* target) noexcept {            \
    applyGate<mlir::qco::KEY##Op, 1>(p0, p1, target);                          \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(                                    \
      double p0, double p1, Qubit* control, Qubit* target) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 1>(p0, p1, control, target);                 \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, double p1,              \
                                            Qubit* control0, Qubit* control1,  \
                                            Qubit* target) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 1>(p0, p1, control0, control1, target);      \
  }
#define MQT_QIR_DEFINE_1_3(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, double p1, double p2,     \
                                          Qubit* target) noexcept {            \
    applyGate<mlir::qco::KEY##Op, 1>(p0, p1, p2, target);                      \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, double p1, double p2,    \
                                           Qubit* control,                     \
                                           Qubit* target) noexcept {           \
    applyGate<mlir::qco::KEY##Op, 1>(p0, p1, p2, control, target);             \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, double p1, double p2,   \
                                            Qubit* control0, Qubit* control1,  \
                                            Qubit* target) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 1>(p0, p1, p2, control0, control1, target);  \
  }
#define MQT_QIR_DEFINE_2_0(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit* target0,                      \
                                          Qubit* target1) noexcept {           \
    applyGate<mlir::qco::KEY##Op, 2>(target0, target1);                        \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit* control, Qubit* target0,     \
                                           Qubit* target1) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 2>(control, target0, target1);               \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit* control0, Qubit* control1,  \
                                            Qubit* target0,                    \
                                            Qubit* target1) noexcept {         \
    applyGate<mlir::qco::KEY##Op, 2>(control0, control1, target0, target1);    \
  }
#define MQT_QIR_DEFINE_2_1(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(double p0, Qubit* target0,           \
                                          Qubit* target1) noexcept {           \
    applyGate<mlir::qco::KEY##Op, 2>(p0, target0, target1);                    \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(                                    \
      double p0, Qubit* control, Qubit* target0, Qubit* target1) noexcept {    \
    applyGate<mlir::qco::KEY##Op, 2>(p0, control, target0, target1);           \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(double p0, Qubit* control0,        \
                                            Qubit* control1, Qubit* target0,   \
                                            Qubit* target1) noexcept {         \
    applyGate<mlir::qco::KEY##Op, 2>(p0, control0, control1, target0,          \
                                     target1);                                 \
  }
#define MQT_QIR_DEFINE_2_2(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(                                     \
      double p0, double p1, Qubit* target0, Qubit* target1) noexcept {         \
    applyGate<mlir::qco::KEY##Op, 2>(p0, p1, target0, target1);                \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(double p0, double p1,               \
                                           Qubit* control, Qubit* target0,     \
                                           Qubit* target1) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 2>(p0, p1, control, target0, target1);       \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(                                   \
      double p0, double p1, Qubit* control0, Qubit* control1, Qubit* target0,  \
      Qubit* target1) noexcept {                                               \
    applyGate<mlir::qco::KEY##Op, 2>(p0, p1, control0, control1, target0,      \
                                     target1);                                 \
  }
#define MQT_QIR_DEFINE_3_0(KEY, NAME, SUFFIX)                                  \
  void __quantum__qis__##NAME##__##SUFFIX(Qubit* target0, Qubit* target1,      \
                                          Qubit* target2) noexcept {           \
    applyGate<mlir::qco::KEY##Op, 3>(target0, target1, target2);               \
  }                                                                            \
  void __quantum__qis__c##NAME##__##SUFFIX(Qubit* control, Qubit* target0,     \
                                           Qubit* target1,                     \
                                           Qubit* target2) noexcept {          \
    applyGate<mlir::qco::KEY##Op, 3>(control, target0, target1, target2);      \
  }                                                                            \
  void __quantum__qis__cc##NAME##__##SUFFIX(Qubit* control0, Qubit* control1,  \
                                            Qubit* target0, Qubit* target1,    \
                                            Qubit* target2) noexcept {         \
    applyGate<mlir::qco::KEY##Op, 3>(control0, control1, target0, target1,     \
                                     target2);                                 \
  }
#define MQT_QIR_DEFINE_CTL_1_0(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Qubit* target) noexcept {        \
    applyControlled<mlir::qco::KEY##Op>(controls, target);                     \
  }
#define MQT_QIR_DEFINE_CTL_1_1(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 1, 1>(controls, args);            \
  }
#define MQT_QIR_DEFINE_CTL_1_2(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 2, 1>(controls, args);            \
  }
#define MQT_QIR_DEFINE_CTL_1_3(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 3, 1>(controls, args);            \
  }
#define MQT_QIR_DEFINE_CTL_2_0(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 0, 2>(controls, args);            \
  }
#define MQT_QIR_DEFINE_CTL_2_1(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 1, 2>(controls, args);            \
  }
#define MQT_QIR_DEFINE_CTL_2_2(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 2, 2>(controls, args);            \
  }
#define MQT_QIR_DEFINE_CTL_3_0(KEY, NAME, CTL_SUFFIX)                          \
  void __quantum__qis__##NAME##__##CTL_SUFFIX(Array* controls,                 \
                                              Tuple* args) noexcept {          \
    applyControlledTuple<mlir::qco::KEY##Op, 0, 3>(controls, args);            \
  }

#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  MQT_QIR_DEFINE_##TARGETS##_##PARAMS(KEY, NAME, SUFFIX)                       \
      MQT_QIR_DEFINE_CTL_##TARGETS##_##PARAMS(KEY, NAME, CTL_SUFFIX)
#include "mqt/Conversion/GateTable.def"

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

void __quantum__qis__gphase__body(const double phase) noexcept {
  qir::Runtime::getInstance().applyGlobalPhase(phase);
}

void __quantum__qis__cnot__body(Qubit* control, Qubit* target) noexcept {
  __quantum__qis__cx__body(control, target);
}

void __quantum__qis__mz__body(Qubit* qubit, Result* result) noexcept {
  return runtimeCall([&] {
    auto& runtime = qir::Runtime::getInstance();
    runtime.measure(qubit, result);
  });
}

void __quantum__qis__reset__body(Qubit* qubit) noexcept {
  return runtimeCall([&] {
    auto& runtime = qir::Runtime::getInstance();
    runtime.reset(std::array{qubit});
  });
}

void __quantum__rt__initialize(char* /*unused*/) noexcept {
  qir::Runtime::getInstance().reset();
}

bool __quantum__rt__read_result(Result* result) noexcept {
  return runtimeCall([&] {
    auto& runtime = qir::Runtime::getInstance();
    return runtime.deref(result).r;
  });
}

void __quantum__rt__result_record_output(Result* result,
                                         const char* label) noexcept {
  const bool bit = __quantum__rt__read_result(result);
  auto& runtime = qir::Runtime::getInstance();
  runtime.outputResult(bit, label);
  // Accumulate new measurement bit.
  runtime.appendMeasurementBit(bit);
}

void __quantum__rt__bool_record_output(bool value, const char* label) noexcept {
  auto& runtime = qir::Runtime::getInstance();
  runtime.outputBool(value, label);
  runtime.appendMeasurementBit(value);
}

void __quantum__rt__int_record_output(int64_t value,
                                      const char* label) noexcept {
  qir::Runtime::getInstance().outputInt(value, label);
}

void __quantum__rt__double_record_output(double value,
                                         const char* label) noexcept {
  qir::Runtime::getInstance().outputFloat(value, label);
}

void __quantum__rt__tuple_record_output(int64_t elementCount,
                                        const char* label) noexcept {
  qir::Runtime::getInstance().outputTuple(elementCount, label);
}

void __quantum__rt__array_record_output(int64_t size,
                                        const char* label) noexcept {
  qir::Runtime::getInstance().outputArray(size, label);
}

void __quantum__rt__result_array_record_output(const int64_t size,
                                               Result** results,
                                               const char* label) noexcept {
  if (size < 0 || (size > 0 && results == nullptr)) {
    fail("Invalid QIR result array output");
  }
  auto& runtime = qir::Runtime::getInstance();
  std::string values;
  if (runtime.hasOutput()) {
    values.reserve(static_cast<std::size_t>(size));
  }
  for (Result* result : std::span(results, static_cast<std::size_t>(size))) {
    const auto value = __quantum__rt__read_result(result);
    if (runtime.hasOutput()) {
      values.push_back(value ? '1' : '0');
    }
    runtime.appendMeasurementBit(value);
  }
  runtime.outputResultArray(values, label);
}

} // extern "C"
