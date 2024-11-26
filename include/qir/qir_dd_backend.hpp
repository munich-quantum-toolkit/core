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
struct StringImpl {
  uint64_t refcount;
  std::string content;
  friend auto operator<<(std::ostream& os,
                         const StringImpl& s) -> std::ostream& {
    return os << s.content;
  }
};
struct TupleImpl {
  uint64_t refcount;
  // todo
};
struct ArrayImpl {
  uint64_t refcount;
  uint64_t aliasCount;
  std::vector<int8_t> data;
  uint32_t elementSize;
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
public:
  static constexpr auto* RESULT_ZERO_ADDRESS =
      reinterpret_cast<Result*>(0x10000);
  static constexpr auto* RESULT_ONE_ADDRESS =
      reinterpret_cast<Result*>(0x10001);

private:
  static constexpr auto MIN_DYN_QUBIT_ADDRESS =
      reinterpret_cast<Qubit*>(0x10000);
  enum class AddressMode : uint8_t { UNKNOWN, DYNAMIC, STATIC };

  AddressMode addressMode;
  std::unordered_map<Qubit*, qc::Qubit> qRegister;
  static constexpr auto MIN_DYN_RESULT_ADDRESS =
      reinterpret_cast<Result*>(0x10000);
  std::unordered_map<Result*, Result> rRegister;
  Qubit* currentMaxQubitAddress;
  qc::Qubit currentMaxQubitId;
  Result* currentMaxResultAddress;
  uint64_t numQubitsInQState;
  dd::Package<> dd;
  dd::vEdge qState;
  std::mt19937_64 mt;

  QIR_DD_Backend();
  explicit QIR_DD_Backend(uint64_t randomSeed);

  [[nodiscard]] static auto generateRandomSeed() -> uint64_t;
  template <typename... Args>
  auto translateAddresses(const Args... qubits) -> std::vector<qc::Qubit>;
  template <typename... Params, typename... Args>
  auto createOperation(qc::OpType op, const Params... params,
                       const Args... qubits) const -> qc::StandardOperation;
  auto enlargeState(const qc::StandardOperation& operation) -> void;

public:
  static QIR_DD_Backend& getInstance() {
    static QIR_DD_Backend instance;
    return instance;
  }

  QIR_DD_Backend(const QIR_DD_Backend&) = delete;
  QIR_DD_Backend& operator=(const QIR_DD_Backend&) = delete;

  template <typename... Params, typename... Args>
  auto apply(qc::OpType op, const Params... params,
             const Args... qubits) -> void;
  template <typename... Args, typename... Results>
  auto measure(const Args... qubits, Results... results) -> void;
  auto qAlloc() -> Qubit*;
  auto qFree(Qubit* qubit) -> void;

  auto rAlloc() -> Result*;
  auto deref(Result* result) -> Result&;
  auto rFree(Result* result) -> void;
  auto equal(Result* result1, Result* result2) -> bool;
};
} // namespace mqt
