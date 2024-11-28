#pragma once

#include "Definitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qir.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct BigIntImpl {
  int32_t refcount;
  int64_t i;
};
/// @note this struct is purposefully not called ResultImpl to leave the Result
/// pointer opaque such that it cannot be dereferenced
struct ResultStruct {
  int32_t refcount;
  bool r;
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

class Utils {
private:
  template <typename Func, typename Store, typename S, typename R,
            std::size_t... I>
  constexpr static void
  transformStoreImpl(Func&& func, Store&& store,
                     const std::array<S, sizeof...(I)>& source,
                     std::array<R, sizeof...(I)>& result,
                     std::index_sequence<I...> /*anonymous*/) {
    ((std::forward<Store>(store)(result[I],
                                 std::forward<Func>(func)(source[I]))),
     ...);
  }

public:
  /// Helper function to apply a function to each element of the array and store
  /// the result in another equally sized array.
  template <typename Func, typename S, typename R, std::size_t N>
  constexpr static void transform(Func&& func, const std::array<S, N>& source,
                                  std::array<R, N>& result) {
    transformStoreImpl(
        std::forward<Func>(func),
        [](auto& container, const auto value) { container = value; }, source,
        result, std::make_index_sequence<N>{});
  }
  /// Helper function to apply a function to each element of the array and store
  /// the result with the help of the store function in another equally sized
  /// array.
  template <typename Func, typename Store, typename S, typename R,
            std::size_t N>
  constexpr static void transformStore(Func&& func, Store&& store,
                                       const std::array<S, N>& source,
                                       std::array<R, N>& result) {
    transformStoreImpl(std::forward<Func>(func), std::forward<Store>(store),
                       source, result, std::make_index_sequence<N>{});
  }
};
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
  std::unordered_map<const Qubit*, qc::Qubit> qRegister;
  static constexpr auto MIN_DYN_RESULT_ADDRESS = 0x10000;
  std::unordered_map<Result*, ResultStruct> rRegister;
  uintptr_t currentMaxQubitAddress;
  qc::Qubit currentMaxQubitId;
  uintptr_t currentMaxResultAddress;
  uint64_t numQubitsInQState;
  std::unique_ptr<dd::Package<>> dd;
  dd::vEdge qState;
  std::mt19937_64 mt;

  QIR_DD_Backend();
  explicit QIR_DD_Backend(uint64_t randomSeed);

  template <size_t P_NUM, size_t SIZE>
  auto createOperation(qc::OpType op, std::array<double, P_NUM> params,
                       std::array<const Qubit*, SIZE> qubits)
      -> qc::StandardOperation;
  auto enlargeState(std::uint64_t maxQubit) -> void;

public:
  [[nodiscard]] static auto generateRandomSeed() -> uint64_t;
  static QIR_DD_Backend& getInstance(const bool reinitialized = false) {
    static QIR_DD_Backend instance;
    if (reinitialized) {
      instance.addressMode = AddressMode::UNKNOWN;
      instance.currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
      instance.currentMaxQubitId = 0;
      instance.currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
      instance.numQubitsInQState = 0;
      instance.dd->decRef(instance.qState);
      instance.qState = dd::vEdge::one();
      instance.dd->garbageCollect();
      instance.mt.seed(generateRandomSeed());
      instance.qRegister.clear();
      instance.rRegister.clear();
      // NOLINTBEGIN(performance-no-int-to-ptr)
      instance.rRegister.emplace(reinterpret_cast<Result*>(RESULT_ZERO_ADDRESS),
                                 ResultStruct{0, false});
      instance.rRegister.emplace(reinterpret_cast<Result*>(RESULT_ONE_ADDRESS),
                                 ResultStruct{0, true});
      // NOLINTEND(performance-no-int-to-ptr)
    }
    return instance;
  }

  QIR_DD_Backend(const QIR_DD_Backend&) = delete;
  QIR_DD_Backend& operator=(const QIR_DD_Backend&) = delete;
  QIR_DD_Backend(QIR_DD_Backend&&) = delete;
  QIR_DD_Backend& operator=(QIR_DD_Backend&&) = delete;

  template <size_t P_NUM, size_t SIZE>
  auto apply(qc::OpType op, std::array<double, P_NUM> params,
             std::array<const Qubit*, SIZE> qubits) -> void;
  template <size_t SIZE>
  auto apply(const qc::OpType op,
             std::array<const Qubit*, SIZE> qubits) -> void {
    apply<0, SIZE>(op, {}, qubits);
  }
  template <size_t SIZE>
  auto measure(std::array<const Qubit*, SIZE> qubits,
               std::array<Result*, SIZE> results) -> void;
  template <size_t SIZE>
  auto reset(std::array<const Qubit*, SIZE> qubits) -> void;
  auto qAlloc() -> Qubit*;
  auto qFree(Qubit* qubit) -> void;
  template <size_t SIZE>
  auto translateAddresses(std::array<const Qubit*, SIZE> qubits)
      -> std::array<qc::Qubit, SIZE>;

  auto rAlloc() -> Result*;
  auto deref(Result* result) -> ResultStruct&;
  auto rFree(Result* result) -> void;
  auto equal(Result* result1, Result* result2) -> bool;
};
} // namespace mqt
