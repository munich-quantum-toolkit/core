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
#include <type_traits>
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
  template <typename T> constexpr static bool is_std_array_v = false;
  template <typename T, std::size_t N>
  constexpr static bool is_std_array_v<std::array<T, N>> = true;
  template <typename Func, typename S, typename T, size_t... I>
  constexpr static void
  apply2Impl(Func&& func, S&& arg1, T&& arg2,
             [[maybe_unused]] std::index_sequence<I...> _) {
    ((std::forward<Func>(func)(std::forward<S>(arg1)[I],
                               std::forward<T>(arg2)[I])),
     ...);
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
      return std::make_tuple(head, tail...);
    } else {
      return skipFirstNArgs<N - 1>(tail...);
    }
  }

public:
  /// Helper function to apply a function to each element of the array and store
  /// the result in another equally sized array.
  template <typename Func, typename S, typename R>
  constexpr static void transform(Func&& func, S&& source, R&& result) {
    static_assert(!std::is_const_v<R>, "Result array must not be const");
    apply2(
        [&func](auto value, auto&& container) {
          container =
              std::forward<Func>(func)(std::forward<decltype(value)>(value));
        },
        std::forward<S>(source), std::forward<R>(result));
  }
  /// Helper function to apply a function to each element of the array and store
  /// the result with the help of the store function in another equally sized
  /// array.
  template <typename Func, typename S, typename T>
  constexpr static void apply2(Func&& func, S&& arg1, T&& arg2) {
    static_assert(
        is_std_array_v<std::remove_const_t<std::remove_reference_t<S>>>,
        "Second argument must be an array");
    static_assert(
        is_std_array_v<std::remove_const_t<std::remove_reference_t<T>>>,
        "Third argument must be an array");
    static_assert(
        std::extent_v<std::remove_const_t<std::remove_reference_t<S>>> ==
            std::extent_v<std::remove_const_t<std::remove_reference_t<T>>>,
        "Both arrays must have the same size");
    constexpr auto n =
        std::extent_v<std::remove_const_t<std::remove_reference_t<S>>>;
    apply2Impl(std::forward<Func>(func), std::forward<S>(arg1),
               std::forward<T>(arg2), std::make_index_sequence<n>{});
  }
  /// To retrieve the first N arguments of a variadic template pack as an array.
  template <size_t N, typename T, typename... Args>
  constexpr static std::array<T, N> getFirstNArgs(Args... args) {
    static_assert(sizeof...(args) >= N, "Not enough arguments provided");
    std::array<T, N> arr = {};
    if constexpr (N > 0) {
      fillArray<0>(arr, args...);
    }
    return arr;
  }
  /// To skip the first M arguments of a variadic template pack and retrieve the
  /// following N arguments as an array.
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

  template <typename... Args>
  auto createOperation(qc::OpType op, Args&... args) -> qc::StandardOperation;
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

  template <typename... Args> auto apply(qc::OpType op, Args&&... args) -> void;
  template <typename... Args> auto measure(Args... args) -> void;
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
