#include "qir/qir_dd_backend.hpp"

#include "dd/Operations.hpp"
#include "qir/qir.h"

#include <stdexcept>

namespace mqt {

auto QIR_DD_Backend::generateRandomSeed() -> uint64_t {
  std::array<std::random_device::result_type, std::mt19937_64::state_size>
      randomData{};
  std::random_device rd;
  std::generate(randomData.begin(), randomData.end(), std::ref(rd));
  std::seed_seq seeds(randomData.begin(), randomData.end());
  std::mt19937_64 rng(seeds);
  return rng();
}

QIR_DD_Backend::QIR_DD_Backend() : QIR_DD_Backend(generateRandomSeed()) {}

QIR_DD_Backend::QIR_DD_Backend(const uint64_t randomSeed)
    : addressMode(AddressMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), numQubitsInQState(0),
      qState(dd::vEdge::one()), mt(randomSeed) {
  dd.incRef(qState);
  qRegister = std::unordered_map<const Qubit*, qc::Qubit>();
  rRegister = std::unordered_map<Result*, ResultStruct>();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ZERO_ADDRESS),
                    ResultStruct{0, false});
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ONE_ADDRESS),
                    ResultStruct{0, true});
  // NOLINTEND(performance-no-int-to-ptr)
}

template <size_t SIZE>
auto QIR_DD_Backend::translateAddresses(std::array<const Qubit*, SIZE> qubits)
    -> std::array<qc::Qubit, SIZE> {
  // extract addresses from opaque qubit pointers
  std::array<qc::Qubit, SIZE> qubitIds;
  if (addressMode != AddressMode::STATIC) {
    // addressMode == AddressMode::DYNAMIC or AddressMode::UNKNOWN
    for (size_t i = 0; i < SIZE; ++i) {
      try {
        qubitIds[i] = qRegister.at(qubits[i]);
      } catch (const std::out_of_range&) {
        if (addressMode == AddressMode::DYNAMIC) {
          std::stringstream ss;
          ss << "Qubit not allocated (not found): " << qubits[i];
          throw std::out_of_range(ss.str());
        }
        // addressMode == AddressMode::UNKNOWN
        addressMode = AddressMode::STATIC;
        break;
      }
    }
  }
  // addressMode might have changed to STATIC
  if (addressMode == AddressMode::STATIC) {
    for (size_t i = 0; i < SIZE; ++i) {
      qubitIds[i] =
          static_cast<qc::Qubit>(reinterpret_cast<uintptr_t>(qubits[i]));
    }
  }
  return qubitIds;
}

template <size_t P_NUM, size_t SIZE>
auto QIR_DD_Backend::createOperation(
    qc::OpType op, std::array<double, P_NUM> params,
    std::array<const Qubit*, SIZE> qubits) -> qc::StandardOperation {
  const auto& addresses = translateAddresses(qubits);
  // store parameters into vector
  std::vector<qc::fp> paramVec(params.cbegin(), params.cend());
  // split addresses into control and target
  uint8_t t = 0;
  if (isSingleQubitGate(op)) {
    t = 1;
  } else if (isTwoQubitGate(op)) {
    t = 2;
  } else {
    throw std::invalid_argument("Operation type is not known: " + toString(op));
  }
  if (addresses.size() > t) { // create controlled operation
    const auto& controls =
        qc::Controls(addresses.cbegin(), addresses.cend() - t);
    const auto& targets = qc::Targets(addresses.cbegin() + t, addresses.cend());
    return {controls, targets, op, paramVec};
  }
  if (addresses.size() == t) { // create uncontrolled operation
    const auto& targets = qc::Targets(addresses.cbegin(), addresses.cend());
    return {targets, op, paramVec};
  }
  throw std::invalid_argument("Operation requires more qubits than given (" +
                              toString(op) +
                              "): " + std::to_string(addresses.size()));
}

auto QIR_DD_Backend::enlargeState(const qc::StandardOperation& operation)
    -> void {
  const auto& qubits = operation.getUsedQubits();
  const auto maxTarget = *std::max_element(qubits.cbegin(), qubits.cend());
  if (const auto d = maxTarget - numQubitsInQState + 1; d > 0) {
    dd.decRef(qState);
    qState = dd.kronecker(qState, dd.makeZeroState(d), d);
    dd.incRef(qState);
    numQubitsInQState += d;
  }
}

template <size_t P_NUM, size_t SIZE>
auto QIR_DD_Backend::apply(const qc::OpType op,
                           std::array<double, P_NUM> params,
                           std::array<const Qubit*, SIZE> qubits) -> void {
  const auto& operation = createOperation<P_NUM, SIZE>(op, params, qubits);
  const auto& matrix = getDD(&operation, dd);
  enlargeState(operation);
  // apply operation
  const auto& tmp = dd.multiply(matrix, qState);
  dd.incRef(tmp);
  dd.decRef(qState);
  qState = tmp;
  dd.garbageCollect();
}

template <size_t SIZE>
auto QIR_DD_Backend::measure(std::array<const Qubit*, SIZE> qubits,
                             std::array<Result*, SIZE> results) -> void {
  auto& backend = QIR_DD_Backend::getInstance();
  const qc::StandardOperation& measure =
      createOperation<0, SIZE>(qc::Measure, {}, qubits);
  enlargeState(measure);
  // measure qubits
  for (size_t i = 0; i < SIZE; ++i) {
    const auto q = static_cast<dd::Qubit>(measure.getTargets()[i]);
    const auto& result = dd.measureOneCollapsing(qState, q, true, mt);
    backend.deref(results[i]).r = result == '1';
  }
}

auto QIR_DD_Backend::qAlloc() -> Qubit* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  qRegister.emplace(qubit, currentMaxQubitId++);
  return qubit;
}

auto QIR_DD_Backend::qFree(Qubit* qubit) -> void {
  apply<1>(qc::Reset, {qubit});
  qRegister.erase(qubit);
}

auto QIR_DD_Backend::rAlloc() -> Result* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{1, false});
  return result;
}

auto QIR_DD_Backend::rFree(Result* result) -> void { rRegister.erase(result); }

auto QIR_DD_Backend::deref(Result* result) -> ResultStruct& {
  auto it = rRegister.find(result);
  if (it == rRegister.end()) {
    if (addressMode != AddressMode::UNKNOWN) {
      addressMode = AddressMode::STATIC;
    }
    if (addressMode == AddressMode::DYNAMIC) {
      std::stringstream ss;
      ss << "Result not allocated (not found): " << result;
      throw std::out_of_range(ss.str());
    }
    it = rRegister.emplace(result, ResultStruct{0, false}).first;
  }
  return it->second;
}

auto QIR_DD_Backend::equal(Result* result1, Result* result2) -> bool {
  return deref(result1).r == deref(result2).r;
}

} // namespace mqt

extern "C" {

// *** MEASUREMENT RESULTS ***
Result* __quantum__rt__result_get_zero() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Result*>(mqt::QIR_DD_Backend::RESULT_ZERO_ADDRESS);
}

Result* __quantum__rt__result_get_one() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<Result*>(mqt::QIR_DD_Backend::RESULT_ONE_ADDRESS);
}

bool __quantum__rt__result_equal(Result* result1, Result* result2) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  return backend.equal(result1, result2);
}

void __quantum__rt__result_update_reference_count(Result* result,
                                                  const int32_t k) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  if (result != nullptr) {
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
  case PauliX:
    string->content = "PauliX";
  case PauliZ:
    string->content = "PauliZ";
  case PauliY:
    string->content = "PauliY";
  }
  return string;
}

String* __quantum__rt__qubit_to_string(const Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  return __quantum__rt__int_to_string(
      static_cast<int32_t>(backend.translateAddresses<1>({qubit}).front()));
}

String* __quantum__rt__range_to_string(Range /*unused*/) {
  throw std::bad_function_call();
}

String* __quantum__rt__bigint_to_string(BigInt* /*unused*/) {
  throw std::bad_function_call();
}

// *** BIG INTEGERS ***
BigInt* __quantum__rt__bigint_create_i64(int64_t /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_create_array(int32_t /*unused*/,
                                           int8_t* /*unused*/) {
  throw std::bad_function_call();
}

int8_t* __quantum__rt__bigint_get_data(BigInt* /*unused*/) {
  throw std::bad_function_call();
}

int32_t __quantum__rt__bigint_get_length(BigInt* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__bigint_update_reference_count(BigInt* /*unused*/,
                                                  int32_t /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_negate(BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_add(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_subtract(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_multiply(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_divide(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_modulus(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_power(BigInt* /*unused*/, int32_t /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_bitand(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_bitor(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_bitxor(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_bitnot(BigInt* /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_shiftleft(BigInt* /*unused*/,
                                        int64_t /*unused*/) {
  throw std::bad_function_call();
}

BigInt* __quantum__rt__bigint_shiftright(BigInt* /*unused*/,
                                         int64_t /*unused*/) {
  throw std::bad_function_call();
}

bool __quantum__rt__bigint_equal(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

bool __quantum__rt__bigint_greater(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

bool __quantum__rt__bigint_greater_eq(BigInt* /*unused*/, BigInt* /*unused*/) {
  throw std::bad_function_call();
}

// *** TUPLES ***
Tuple* __quantum__rt__tuple_create(int64_t /*unused*/) {
  throw std::bad_function_call();
}

Tuple* __quantum__rt__tuple_copy(Tuple* /*unused*/, bool /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__tuple_update_reference_count(Tuple* /*unused*/,
                                                 int32_t /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__tuple_update_alias_count(Tuple* /*unused*/,
                                             int32_t /*unused*/) {
  throw std::bad_function_call();
}

// *** ARRAYS ***
Array* __quantum__rt__array_create_1d(const int32_t size, const int64_t n) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* array = new Array;
  array->refcount = 1;
  array->aliasCount = 0;
  array->data =
      std::vector(static_cast<size_t>(size * n), static_cast<int8_t>(0));
  array->elementSize = n;
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
  }
  array->refcount++;
  return array;
}

Array* __quantum__rt__array_concatenate(Array* /*unused*/, Array* /*unused*/) {
  throw std::bad_function_call();
}

Array* __quantum__rt__array_slice_1d(Array* /*unused*/, Range /*unused*/,
                                     bool /*unused*/) {
  throw std::bad_function_call();
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
  throw std::bad_function_call();
}

int32_t __quantum__rt__array_get_dim(Array* /*unused*/) {
  throw std::bad_function_call();
}

int64_t __quantum__rt__array_get_size(Array* /*unused*/, int32_t /*unused*/) {
  throw std::bad_function_call();
}

int8_t* __quantum__rt__array_get_element_ptr(Array* /*unused*/,
                                             int64_t* /*unused*/) {
  throw std::bad_function_call();
}

Array* __quantum__rt__array_slice(Array* /*unused*/, int32_t /*unused*/,
                                  Range /*unused*/, bool /*unused*/) {
  throw std::bad_function_call();
}

Array* __quantum__rt__array_project(Array* /*unused*/, int32_t /*unused*/,
                                    int64_t /*unused*/, bool /*unused*/) {
  throw std::bad_function_call();
}

// *** CALLABLES ***
Callable*
__quantum__rt__callable_create(void (* /*unused*/[4])(Tuple*, Tuple*, Tuple*),
                               void (* /*unused*/[2])(Tuple*, Tuple*, Tuple*),
                               Tuple* /*unused*/) {
  throw std::bad_function_call();
}

Callable* __quantum__rt__callable_copy(Callable* /*unused*/, bool /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__callable_invoke(Callable* /*unused*/, Tuple* /*unused*/,
                                    Tuple* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__callable_make_adjoint(Callable* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__callable_make_controlled(Callable* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__callable_update_reference_count(Callable* /*unused*/,
                                                    const int32_t /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__callable_update_alias_count(Callable* /*unused*/,
                                                const int32_t /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__capture_update_reference_count(Callable* /*unused*/,
                                                   const int32_t /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__capture_update_alias_count(Callable* /*unused*/,
                                               const int32_t /*unused*/) {
  throw std::bad_function_call();
}

// *** CLASSICAL RUNTIME ***
void __quantum__rt__message(const String* msg) { std::cout << *msg << "\n"; }

void __quantum__rt__fail(const String* msg) {
  throw std::runtime_error(msg->content);
}

// *** QUANTUM INSTRUCTION SET AND RUNTIME ***
Qubit* __quantum__rt__qubit_allocate() {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  return backend.qAlloc();
}

Array* __quantum__rt__qubit_allocate_array(const int64_t n) {
  auto* array = __quantum__rt__array_create_1d(sizeof(Qubit*), n);
  for (int64_t i = 0; i < n; ++i) {
    *static_cast<Qubit**>(
        // NOLINTNEXTLINE(bugprone-casting-through-void)
        static_cast<void*>(__quantum__rt__array_get_element_ptr_1d(array, i))) =
        __quantum__rt__qubit_allocate();
  }
  return array;
}

void __quantum__rt__qubit_release(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.qFree(qubit);
}

void __quantum__rt__qubit_release_array(Array* array) {
  const auto size = __quantum__rt__array_get_size_1d(array);
  // deallocate every qubit
  for (int64_t i = 0; i < size; ++i) {
    __quantum__rt__qubit_release(*static_cast<Qubit**>(
        // NOLINTNEXTLINE(bugprone-casting-through-void)
        static_cast<void*>(__quantum__rt__array_get_element_ptr_1d(array, i))));
  }
  // deallocate array
  __quantum__rt__array_update_reference_count(array, -1);
}

// QUANTUM INSTRUCTION SET
void __quantum__qis__x__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::X, {qubit});
}

void __quantum__qis__y__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::Y, {qubit});
}

void __quantum__qis__z__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::Z, {qubit});
}

void __quantum__qis__h__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::H, {qubit});
}

void __quantum__qis__sqrtx__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::SX, {qubit});
}

void __quantum__qis__sqrtxdg__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::SXdg, {qubit});
}

void __quantum__qis__s__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::S, {qubit});
}

void __quantum__qis__sdg__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::Sdg, {qubit});
}

void __quantum__qis__t__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::T, {qubit});
}

void __quantum__qis__tdg__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::Tdg, {qubit});
}

void __quantum__qis__rx__body(const double phi, Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1, 1>(qc::RX, {phi}, {qubit});
}

void __quantum__qis__ry__body(double phi, Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1, 1>(qc::RY, {phi}, {qubit});
}

void __quantum__qis__rz__body(double phi, Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1, 1>(qc::RZ, {phi}, {qubit});
}

void __quantum__qis__cnot__body(Qubit* control, Qubit* target) {
  __quantum__qis__cx__body(control, target);
}

void __quantum__qis__cx__body(Qubit* control, Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<2>(qc::X, {control, target});
}

void __quantum__qis__cz__body(Qubit* control, Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<2>(qc::Z, {control, target});
}

void __quantum__qis__ccx__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<3>(qc::X, {control1, control2, target});
}

void __quantum__qis__ccz__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<3>(qc::Z, {control1, control2, target});
}

void __quantum__qis__mz__body(Qubit* qubit, Result* result) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.measure<1>({qubit}, {result});
}

Result* __quantum__qis__m__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  auto* result = backend.rAlloc();
  __quantum__qis__mz__body(qubit, result);
  return result;
}

Result* __quantum__qis__measure__body(Qubit* qubit) {
  return __quantum__qis__m__body(qubit);
}

void __quantum__qis__reset__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply<1>(qc::Reset, {qubit});
}

void __quantum__rt__initialize(char* /*unused*/) {
  mqt::QIR_DD_Backend::getInstance();
}

bool __quantum__rt__read_result(Result* result) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  return backend.deref(result).r;
}

void __quantum__rt__tuple_record_output(int64_t /*unused*/,
                                        const char* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__array_record_output(int64_t /*unused*/,
                                        const char* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__result_record_output(Result* /*unused*/,
                                         const char* /*unused*/) {
  throw std::bad_function_call();
}

void __quantum__rt__bool_record_output(bool /*unused*/,
                                       const char* /*unused*/) {
  throw std::bad_function_call();
}

} // extern "C"
