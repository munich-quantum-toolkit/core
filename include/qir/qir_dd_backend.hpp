#pragma once

#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"
#include "qir.h"

struct BigIntImpl {
  uint64_t refcount;
  int64_t i;
};
struct ResultImpl {
  uint64_t refcount;
  bool r;
};
struct QubitImpl {
  qc::Qubit id;
};
struct TupleImpl {
  uint64_t refcount;
  // todo
};
template <typename T> struct ArrayImpl {
  uint64_t refcount;
  std::vector<T> elements;
};
template <size_t size> struct ArrayElement {
  std::array<uint8_t, size> data;
};
struct CallablImpl {
  uint64_t refcount;
  // todo
};

namespace mqt {

/**
 * @note This class is implemented following the design pattern Singleton in
 * order to access an instance of this class from the C function without having
 * a handle to it.
 */
class QIR_DD_Backend {
private:
  static constexpr auto MIN_DYN_ADDRESS = 0x10000;
  enum class AddressMode : uint8_t { UNKNOWN, DYNAMIC, STATIC };

  AddressMode addressMode;
  std::unordered_map<qc::Qubit, qc::Qubit> qRegister;
  uint64_t currentMaxAddress;
  uint64_t numQubits;
  dd::Package<> dd;
  dd::vEdge qState;

  QIR_DD_Backend();

  auto determineAddressMode() -> void;

public:
  static QIR_DD_Backend& getInstance() {
    static QIR_DD_Backend instance;
    return instance;
  }

  QIR_DD_Backend(const QIR_DD_Backend&) = delete;
  QIR_DD_Backend& operator=(const QIR_DD_Backend&) = delete;

  template <typename... Args> auto apply(qc::OpType op, Args... qubits) -> void;
  auto qAlloc() -> Qubit*;
  auto qFree(const Qubit* q) -> void;
};
} // namespace mqt
