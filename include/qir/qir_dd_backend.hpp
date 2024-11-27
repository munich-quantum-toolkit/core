#pragma once

#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"
#include "qir.h"

struct BigIntImpl {
  int32_t refcount;
  int64_t i;
};
struct ResultImpl {
  int32_t refcount;
  bool r;
};
struct QubitImpl {
  qc::Qubit id;
};
struct StringImpl {
  int32_t refcount;
  std::string content;
  friend auto operator<<(std::ostream& os,
                         const StringImpl& s) -> std::ostream& {
    return os << s.content;
  }
};
struct TupleImpl {
  int32_t refcount;
  // todo
};
struct ArrayImpl {
  int32_t refcount;
  int32_t aliasCount;
  std::vector<int8_t> data;
  int64_t elementSize;
};
struct CallablImpl {
  int32_t refcount;
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
  static constexpr auto RESULT_ZERO_ADDRESS = 0x10000;
  static constexpr auto RESULT_ONE_ADDRESS = 0x10001;

private:
  static constexpr auto MIN_DYN_QUBIT_ADDRESS = 0x10000;
  enum class AddressMode : uint8_t { UNKNOWN, DYNAMIC, STATIC };

  AddressMode addressMode;
  std::unordered_map<Qubit*, qc::Qubit> qRegister;
  static constexpr auto MIN_DYN_RESULT_ADDRESS = 0x10000;
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
  template <size_t SIZE>
  auto translateAddresses(std::array<Qubit*, SIZE> qubits)
      -> std::array<qc::Qubit, SIZE>;
  template <size_t P_NUM, size_t SIZE>
  auto
  createOperation(qc::OpType op, std::array<double, P_NUM> params,
                  std::array<Qubit*, SIZE> qubits) -> qc::StandardOperation;
  auto enlargeState(const qc::StandardOperation& operation) -> void;

public:
  static QIR_DD_Backend& getInstance() {
    static QIR_DD_Backend instance;
    return instance;
  }

  QIR_DD_Backend(const QIR_DD_Backend&) = delete;
  QIR_DD_Backend& operator=(const QIR_DD_Backend&) = delete;

  template <size_t P_NUM, size_t SIZE>
  auto apply(qc::OpType op, std::array<double, P_NUM> params,
             std::array<Qubit*, SIZE> qubits) -> void;
  template <size_t SIZE>
  auto apply(const qc::OpType op, std::array<Qubit*, SIZE> qubits) -> void {
    apply<0, SIZE>(op, {}, qubits);
  }
  template <size_t SIZE>
  auto measure(std::array<Qubit*, SIZE> qubits,
               std::array<Result*, SIZE> results) -> void;
  auto qAlloc() -> Qubit*;
  auto qFree(Qubit* qubit) -> void;

  auto rAlloc() -> Result*;
  auto deref(Result* result) -> Result&;
  auto rFree(Result* result) -> void;
  auto equal(Result* result1, Result* result2) -> bool;
};
} // namespace mqt
