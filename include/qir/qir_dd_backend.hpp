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
#include <tuple>
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
  template <typename Func, typename S, typename T, size_t... I>
  constexpr static void
  apply2Impl(Func&& func, std::array<S, sizeof...(I)>& arg1,
             std::array<T, sizeof...(I)>& arg2,
             [[maybe_unused]] std::index_sequence<I...> _) {
    ((std::forward<Func>(func)(arg1[I], arg2[I])), ...);
  }
  template <typename Func, typename S, typename T, size_t... I>
  constexpr static void
  apply2Impl(Func&& func, const std::array<S, sizeof...(I)>& arg1,
             std::array<T, sizeof...(I)>& arg2,
             [[maybe_unused]] std::index_sequence<I...> _) {
    ((std::forward<Func>(func)(arg1[I], arg2[I])), ...);
  }
  template <size_t I, size_t N, typename T, typename... Args>
  constexpr static void fillArray(std::array<T, N>& arr, T head, Args... tail) {
    arr[I] = head;
    if constexpr (N - I > 1) {
      fillArray<I + 1>(arr, tail...);
    }
  }
  template <size_t N, typename T, typename... Args>
  constexpr static auto skipFirstNArgs(T head, Args... tail) {
    if constexpr (N == 0) {
      return std::make_tuple<>(head, tail...);
    } else {
      return skipFirstNArgs<N - 1>(tail...);
    }
  }

public:
  /// Helper function to apply a function to each element of the array and store
  /// the result in another equally sized array.
  template <typename Func, typename S, typename R, size_t N>
  constexpr static void transform(Func&& func, std::array<S, N>& source,
                                  std::array<R, N>& result) {
    apply2(
        [&func](auto value, auto&& container) {
          container =
              std::forward<Func>(func)(std::forward<decltype(value)>(value));
        },
        source, result);
  }
  /// Helper function to apply a function to each element of the array and store
  /// the result with the help of the store function in another equally sized
  /// array.
  template <typename Func, typename S, typename T, size_t N>
  constexpr static void apply2(Func&& func, std::array<S, N>& arg1,
                               std::array<T, N>& arg2) {
    apply2Impl(std::forward<Func>(func), arg1, arg2,
               std::make_index_sequence<N>{});
  }

  template <typename Func, typename S, typename T, size_t N>
  constexpr static void apply2(Func&& func, const std::array<S, N>& arg1,
                               std::array<T, N>& arg2) {
    apply2Impl(std::forward<Func>(func), arg1, arg2,
               std::make_index_sequence<N>{});
  }
  // todo: docstring
  template <size_t N, typename T, typename... Args>
  constexpr static std::array<T, N> getFirstNArgs(Args... args) {
    static_assert(sizeof...(args) >= N, "Not enough arguments provided");
    std::array<T, N> arr = {};
    if constexpr (N > 0) {
      fillArray<0>(arr, args...);
    }
    return arr;
  }
  // todo: docstring
  template <size_t M, size_t N, typename T, typename... Args>
  constexpr static std::array<T, N> getNAfterMArgs(Args... args) {
    static_assert(sizeof...(args) >= M + N, "Not enough arguments provided");
    std::array<T, N> arr = {};
    auto argsAfterM = skipFirstNArgs<M>(args...);
    std::apply([&](auto... args2) { fillArray<0>(arr, args2...); }, argsAfterM);
    return arr;
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
  auto
  createOperation(qc::OpType op, std::array<double, P_NUM> params,
                  std::array<Qubit*, SIZE> qubits) -> qc::StandardOperation;
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

  template <typename... Args> auto apply(qc::OpType op, Args... args) -> void;
  template <size_t SIZE>
  auto measure(std::array<Qubit*, SIZE> qubits,
               std::array<Result*, SIZE> results) -> void;
  template <size_t SIZE> auto reset(std::array<Qubit*, SIZE> qubits) -> void;
  auto qAlloc() -> Qubit*;
  auto qFree(Qubit* qubit) -> void;
  template <size_t SIZE>
  auto translateAddresses(std::array<Qubit*, SIZE> qubits)
      -> std::array<qc::Qubit, SIZE>;

  auto rAlloc() -> Result*;
  auto deref(Result* result) -> ResultStruct&;
  auto rFree(Result* result) -> void;
  auto equal(Result* result1, Result* result2) -> bool;
};
} // namespace mqt
