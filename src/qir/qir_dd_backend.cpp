#include "qir/qir_dd_backend.hpp"

#include "dd/Operations.hpp"
#include "qir/qir.h"

#include <exception>
#include <stdexcept>

namespace mqt {

QIR_DD_Backend::QIR_DD_Backend()
    : addressMode(AddressMode::UNKNOWN), currentMaxAddress(0), numQubits(0),
      qState(dd::vEdge::one()) {
  qRegister = std::unordered_map<qc::Qubit, qc::Qubit>();
}

auto QIR_DD_Backend::determineAddressMode() -> void {
  if (addressMode == AddressMode::UNKNOWN) {
    if (qRegister.empty()) {
      addressMode = AddressMode::STATIC;
    } else {
      addressMode = AddressMode::DYNAMIC;
    }
  }
}

template <typename... Args>
auto QIR_DD_Backend::apply(const qc::OpType op, const Args... qubits) -> void {
  determineAddressMode();
  // extract addresses from opaque qubit pointers
  const std::vector<Qubit*> rawAddresses = {qubits...};
  std::vector<qc::Qubit> addresses;
  addresses.reserve(rawAddresses.size());
  if (addressMode == AddressMode::STATIC) {
    std::transform(rawAddresses.cbegin(), rawAddresses.cend(),
                   std::back_inserter(addresses),
                   [](const auto a) { return static_cast<qc::Qubit>(a); });
  } else { // addressMode == AddressMode::DYNAMIC
    std::transform(
        rawAddresses.cbegin(), rawAddresses.cend(),
        std::back_inserter(addresses), [&](const auto a) {
          if (a < MIN_DYN_ADDRESS) {
            std::out_of_range("Qubit not allocated (out of range): " +
                              std::to_string(a));
          }
          try {
            return qRegister.at(a->id);
          } catch (const std::out_of_range&) {
            std::throw_with_nested(std::out_of_range(
                "Qubit not allocated (not found): " + std::to_string(a->id)));
          }
        });
  }
  // split addresses into control and target
  uint8_t t = 0;
  if (isSingleQubitGate(op)) {
    t = 1;
  } else if (isTwoQubitGate(op)) {
    t = 2;
  } else {
    throw std::invalid_argument("Operation type is not known: " + toString(op));
  }
  qc::StandardOperation operation;
  if (addresses.size() > t) { // create controlled operation
    const auto& controls =
        qc::Controls(addresses.cbegin(), addresses.cend() - t);
    const auto& targets = qc::Targets(addresses.cbegin() - t, addresses.cend());
    operation = qc::StandardOperation(controls, targets, op);
  } else if (addresses.size() == t) { // create uncontrolled operation
    operation = qc::StandardOperation(addresses, op);
  } else {
    throw std::invalid_argument("Operation requires more qubits than given (" +
                                qc::toString(op) +
                                "): " + std::to_string(addresses.size()));
  }
  // retrieve operation matrix
  const auto& matrix = getDD(&operation, dd);
  // enlarge quantum state if necessary
  const auto maxTarget =
      *std::max_element(addresses.cbegin(), addresses.cend());
  const auto d = maxTarget - numQubits + 1;
  if (d > 0) {
    qState = dd.kronecker(qState, dd.makeZeroState(d), d);
    numQubits += d;
  }
  // apply operation
  const auto& tmp = dd.multiply(matrix, qState);
  dd.incRef(tmp);
  dd.decRef(qState);
  qState = tmp;
  dd.garbageCollect();
}

auto QIR_DD_Backend::qAlloc() -> Qubit* {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* q = new Qubit;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  q->id = currentMaxAddress + MIN_DYN_ADDRESS;
  qRegister.emplace(q->id, currentMaxAddress);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  ++currentMaxAddress;
  return q;
}

auto QIR_DD_Backend::qFree(const Qubit* q) -> void {
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  apply(qc::Reset, q);
  qRegister.erase(q->id);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete q;
}

} // namespace mqt

extern "C" {

// *** MEASUREMENT RESULTS ***
Result* __quantum__rt__result_get_zero() {
  static Result zero = {0, false};
  return &zero;
}

Result* __quantum__rt__result_get_one() {
  static Result one = {0, true};
  return &one;
}

bool __quantum__rt__result_equal(const Result* r1, const Result* r2) {
  if (r1 == nullptr) {
    throw std::invalid_argument("First argument must not be null.");
  }
  if (r2 == nullptr) {
    throw std::invalid_argument("Second argument must not be null.");
  }
  return r1->r == r2->r;
}

void __quantum__rt__result_update_reference_count(Result* r, const int32_t k) {
  if (r != nullptr) {
    r->refcount += k;
    if (r->refcount == 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete r;
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

String* __quantum__rt__result_to_string(const Result* result) {
  if (result == nullptr) {
    throw std::invalid_argument("The argument must not be null.");
  }
  return __quantum__rt__int_to_string(static_cast<int32_t>(result->r));
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
  if (qubit == nullptr) {
    throw std::invalid_argument("The argument must not be null.");
  }
  return __quantum__rt__int_to_string(static_cast<int32_t>(qubit->id));
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
  array->data = std::vector(size * n, static_cast<int8_t>(0));
  array->elementSize = n;
  return array;
}

Array* __quantum__rt__array_copy(Array* array, bool shallow) {
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
  return static_cast<int64_t>(array->data.size() / array->elementSize);
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
  return &array->data[array->elementSize * i];
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

void __quantum__rt__qubit_release(const Qubit* qubit) {
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
  backend.apply(qc::X, qubit);
}

void __quantum__qis__y__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Y, qubit);
}

void __quantum__qis__z__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Z, qubit);
}

void __quantum__qis__h__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::H, qubit);
}

void __quantum__qis__sqrtx__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::SX, qubit);
}

void __quantum__qis__sqrtxdg__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::SXdg, qubit);
}

void __quantum__qis__s__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::S, qubit);
}

void __quantum__qis__sdg__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Sdg, qubit);
}

void __quantum__qis__t__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::T, qubit);
}

void __quantum__qis__tdg__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Tdg, qubit);
}

void __quantum__qis__rx__body(double phi, Qubit* qubit) {
  throw std::bad_function_call();
}

void __quantum__qis__ry__body(double phi, Qubit* qubit) {
  throw std::bad_function_call();
}

void __quantum__qis__rz__body(double phi, Qubit* qubit) {
  throw std::bad_function_call();
}

void __quantum__qis__cnot__body(Qubit* control, Qubit* target) {
  __quantum__qis__cx__body(control, target);
}

void __quantum__qis__cx__body(Qubit* control, Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::X, control, target);
}

void __quantum__qis__cz__body(Qubit* control, Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Z, control, target);
}

void __quantum__qis__ccx__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::X, control1, control2, target);
}

void __quantum__qis__ccz__body(Qubit* control1, Qubit* control2,
                               Qubit* target) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Z, control1, control2, target);
}

void __quantum__qis__mz__body(Qubit* qubit, Result* result) {
  throw std::bad_function_call();
}

void __quantum__qis__m__body(Qubit* qubit, Result* result) {
  __quantum__qis__mz__body(qubit, result);
}

void __quantum__qis__reset__body(Qubit* qubit) {
  auto& backend = mqt::QIR_DD_Backend::getInstance();
  backend.apply(qc::Reset, qubit);
}

} // extern "C"
