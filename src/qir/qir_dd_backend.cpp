#include "qir/qir_dd_backend.hpp"

#include "dd/Operations.hpp"
#include "qir/qir.h"

#include <cstdio>
#include <cstdlib>
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
auto QIR_DD_Backend::apply(const qc::OpType op, Args... qubits) -> void {
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
            std::invalid_argument("Qubit not allocated (out of range): " +
                                  std::to_string(a));
          }
          try {
            return qRegister.at(a->id);
          } catch (const std::out_of_range&) {
            std::throw_with_nested(std::invalid_argument(
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
  const auto& controls = qc::Controls(addresses.cbegin(), addresses.cend() - t);
  const auto& targets = qc::Targets(addresses.cbegin() - t, addresses.cend());
  // retrieve operation matrix
  const auto& operation = qc::StandardOperation(controls, targets, op);
  const auto& matrix = getDD(&operation, dd);
  // enlarge quantum state if necessary
  const auto maxTarget = *std::max_element(targets.cbegin(), targets.cend());
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
String* __quantum__rt__string_create(const char*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

const char* __quantum__rt__string_get_data(String*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

int32_t __quantum__rt__string_get_length(String*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return 0;
}

void __quantum__rt__string_update_reference_count(String*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

String* __quantum__rt__string_concatenate(String*, String*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

bool __quantum__rt__string_equal(String*, String*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return false;
}

String* __quantum__rt__int_to_string(int64_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__double_to_string(double) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__bool_to_string(bool) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__result_to_string(Result*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__pauli_to_string(Pauli) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__qubit_to_string(Qubit*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__range_to_string(Range) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

String* __quantum__rt__bigint_to_string(BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

// *** BIG INTEGERS ***
BigInt* __quantum__rt__bigint_create_i64(int64_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_create_array(int32_t, char*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

char* __quantum__rt__bigint_get_data(BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

int32_t __quantum__rt__bigint_get_length(BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return 0;
}

void __quantum__rt__bigint_update_reference_count(BigInt*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

BigInt* __quantum__rt__bigint_negate(BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_add(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_subtract(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_multiply(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_divide(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_modulus(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_power(BigInt*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_bitand(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_bitor(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_bitxor(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_bitnot(BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_shiftleft(BigInt*, int64_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

BigInt* __quantum__rt__bigint_shiftright(BigInt*, int64_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

bool __quantum__rt__bigint_equal(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

bool __quantum__rt__bigint_greater(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

bool __quantum__rt__bigint_greater_eq(BigInt*, BigInt*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

// *** TUPLES ***
Tuple* __quantum__rt__tuple_create(int64_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Tuple* __quantum__rt__tuple_copy(Tuple*, bool force) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

void __quantum__rt__tuple_update_reference_count(Tuple*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__tuple_update_alias_count(Tuple*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

// *** ARRAYS ***
Array* __quantum__rt__array_create_1d(int32_t size, int64_t n) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  Array* a = malloc(size * n);
  for (int64_t i = 0; i < n; i++) {
    ((char*)a)[i] = '0';
  }
  return a;
}

Array* __quantum__rt__array_copy(Array*, bool) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Array* __quantum__rt__array_concatenate(Array*, Array*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Array* __quantum__rt__array_slice_1d(Array*, Range, bool) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

int64_t __quantum__rt__array_get_size_1d(Array*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return 0;
}

char* __quantum__rt__array_get_element_ptr_1d(Array*, int64_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

void __quantum__rt__array_update_reference_count(Array*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__array_update_alias_count(Array*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

Array* __quantum__rt__array_create(int32_t, int32_t, int64_t*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

int32_t __quantum__rt__array_get_dim(Array*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return 0;
}

int64_t __quantum__rt__array_get_size(Array*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return 0;
}

char* __quantum__rt__array_get_element_ptr(Array*, int64_t*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Array* __quantum__rt__array_slice(Array*, int32_t, Range, bool) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Array* __quantum__rt__array_project(Array*, int32_t, int64_t, bool) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

// *** CALLABLES ***
Callable* __quantum__rt__callable_create(void (*f[4])(Tuple*, Tuple*, Tuple*),
                                         void (*c[2])(Tuple*, Tuple*, Tuple*),
                                         Tuple*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Callable* __quantum__rt__callable_copy(Callable*, bool) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

void __quantum__rt__callable_invoke(Callable*, Tuple*, Tuple*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__callable_make_adjoint(Callable*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__callable_make_controlled(Callable*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__callable_update_reference_count(Callable*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__callable_update_alias_count(Callable*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__capture_update_reference_count(Callable*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__capture_update_alias_count(Callable*, int32_t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

// *** CLASSICAL RUNTIME ***
void __quantum__rt__message(String* msg) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__fail(String* msg) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  exit(1);
}

// *** QUANTUM INSTRUCTION SET AND RUNTIME ***
Qubit* __quantum__rt__qubit_allocate() {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return NULL;
}

Array* __quantum__rt__qubit_allocate_array(int64_t n) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  return malloc(sizeof(Qubit*) * n);
}

void __quantum__rt__qubit_release(Qubit*) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
}

void __quantum__rt__qubit_release_array(Array* a) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  free(a);
}

// QUANTUM INSTRUCTION SET
void __quantum__qis__x__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->x(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__y__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->y(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__z__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->z(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__h__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->h(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__sqrtx__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->sx(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__sqrtxdg__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->sxdg(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__s__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->s(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__sdg__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->sdg(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__t__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->t(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__tdg__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->tdg(reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__rx__body(double phi, Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->rx(phi, reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__ry__body(double phi, Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->ry(phi, reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__rz__body(double phi, Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->rz(phi, reinterpret_cast<qc::Qubit>(q));
}

void __quantum__qis__cnot__body(Qubit* c, Qubit* t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->cx(reinterpret_cast<qc::Qubit>(c), reinterpret_cast<qc::Qubit>(t));
}

void __quantum__qis__cx__body(Qubit* c, Qubit* t) {
  printf("%s:%s:%d \nredirects to: ", __FILE__, __FUNCTION__, __LINE__);
  __quantum__qis__cnot__body(c, t);
}

void __quantum__qis__cz__body(Qubit* c, Qubit* t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->cz(reinterpret_cast<qc::Qubit>(c), reinterpret_cast<qc::Qubit>(t));
}

void __quantum__qis__ccx__body(Qubit* c1, Qubit* c2, Qubit* t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->mcx({reinterpret_cast<qc::Qubit>(c1),
  // reinterpret_cast<qc::Qubit>(c2)}, reinterpret_cast<qc::Qubit>(t));
}

void __quantum__qis__ccz__body(Qubit* c1, Qubit* c2, Qubit* t) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->mcz({reinterpret_cast<qc::Qubit>(c1),
  // reinterpret_cast<qc::Qubit>(c2)}, reinterpret_cast<qc::Qubit>(t));
}

void __quantum__qis__mz__body(Qubit* q, Result* r) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // TODO
}

void __quantum__qis__m__body(Qubit* q, Result* r) {
  printf("%s:%s:%d \nredirects to: ", __FILE__, __FUNCTION__, __LINE__);
  __quantum__qis__mz__body(q, r);
}

void __quantum__qis__reset__body(Qubit* q) {
  printf("%s:%s:%d \n", __FILE__, __FUNCTION__, __LINE__);
  // circ->reset(reinterpret_cast<qc::Qubit>(q));
}

} // extern "C"
