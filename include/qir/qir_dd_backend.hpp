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
// ReSharper disable once CppUnusedIncludeDirective
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

struct BigIntImpl {
  int32_t refcount;
  // todo
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
  int32_t aliasCount;
  std::vector<int8_t> data;
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

// Primary template
template <typename T> static constexpr bool IS_STD_ARRAY_V = false;
// Specialization for std::array
template <typename T, std::size_t N>
static constexpr bool IS_STD_ARRAY_V<std::array<T, N>> = true;

// Primary template
template <typename T, typename... Args> struct SizeOfPackOfType;
// Base case: no matching types
template <typename T> struct SizeOfPackOfType<T> {
  static constexpr size_t VALUE = 0;
};
// Recursive case: first type does not match
template <typename T, typename U, typename... Args>
struct SizeOfPackOfType<T, U, Args...> {
  static constexpr size_t VALUE = SizeOfPackOfType<T>::VALUE;
};
// Recursive case: first type matches
template <typename T, typename... Args> struct SizeOfPackOfType<T, T, Args...> {
  static constexpr size_t VALUE = 1 + SizeOfPackOfType<T, Args...>::VALUE;
};
// Helper variable template
template <typename T, typename... Args>
static inline constexpr size_t SIZE_OF_PACK_OF_TYPE_V =
    SizeOfPackOfType<T, Args...>::VALUE;

// Primary template
template <template <typename, typename...> class V, typename T,
          typename... Args>
struct SkipUntilType;
// Base case: no matching types
template <template <typename, typename...> class V, typename T>
struct SkipUntilType<V, T> {
  static constexpr size_t VALUE = 0;
};
// Recursive case: skip until T is found
template <template <typename, typename...> class V, typename T, typename U,
          typename... Args>
struct SkipUntilType<V, T, U, Args...> {
  static constexpr size_t VALUE = SkipUntilType<V, T, Args...>::VALUE;
};
// Recursive case: T is found
template <template <typename, typename...> class V, typename T,
          typename... Args>
struct SkipUntilType<V, T, T, Args...> {
  static constexpr size_t VALUE = V<T, T, Args...>::VALUE;
};
// Helper type template
template <template <typename, typename...> class V, typename T,
          typename... Args>
static inline constexpr size_t SKIP_UNTIL_TYPE_V =
    SkipUntilType<V, T, Args...>::VALUE;

class Utils {
private:
  template <typename Func, typename S, typename T, size_t... I>
  static constexpr void
  apply2Impl(Func&& func, S&& arg1, T&& arg2,
             [[maybe_unused]] std::index_sequence<I...> _) {
    ((std::forward<Func>(func)(std::forward<S>(arg1)[I],
                               std::forward<T>(arg2)[I])),
     ...);
  }
  template <size_t I, size_t N, typename T, typename... Args>
  static constexpr void fillArray(std::array<T, N>& arr, T head, Args... tail) {
    arr[I] = head;
    if constexpr (N - I > 1) {
      fillArray<I + 1>(arr, tail...);
    }
  }
  template <size_t N, typename T, typename... Args>
  static constexpr auto skipNArgs(T head, Args... tail) {
    if constexpr (N == 0) {
      return std::make_tuple(head, tail...);
    } else {
      return skipNArgs<N - 1>(tail...);
    }
  }
  template <typename T, typename Func, typename S, typename... Args>
  static constexpr auto skipUntilType(Func&& func, S head, Args... tail) {
    if constexpr (std::is_same_v<S, T>) {
      return std::forward<Func>(func)(head, tail...);
    } else {
      static_assert(sizeof...(Args) > 0, "There is no argument of given type.");
      skipUntilType<T>(std::forward<Func>(func), tail...);
    }
  }

public:
  /// Helper function to apply a function to each element of the array and store
  /// the result in another equally sized array.
  template <typename Func, typename S, typename R>
  static constexpr void transform(Func&& func, S&& source, R&& result) {
    static_assert(!std::is_const_v<R>, "Result array must not be const");
    apply2(
        [&func](auto&& value, auto&& container) {
          container =
              std::forward<Func>(func)(std::forward<decltype(value)>(value));
        },
        std::forward<S>(source), std::forward<R>(result));
  }
  /// Helper function to apply a function to each element of the array and store
  /// the result with the help of the store function in another equally sized
  /// array.
  template <typename Func, typename S, typename T>
  static constexpr void apply2(Func&& func, S&& arg1, T&& arg2) {
    static_assert(IS_STD_ARRAY_V<std::remove_cv_t<std::remove_reference_t<S>>>,
                  "Second argument must be an array");
    static_assert(IS_STD_ARRAY_V<std::remove_cv_t<std::remove_reference_t<T>>>,
                  "Third argument must be an array");
    static_assert(
        std::tuple_size_v<std::remove_const_t<std::remove_reference_t<S>>> ==
            std::tuple_size_v<std::remove_const_t<std::remove_reference_t<T>>>,
        "Both arrays must have the same size");
    constexpr auto n =
        std::tuple_size_v<std::remove_cv_t<std::remove_reference_t<S>>>;
    apply2Impl(std::forward<Func>(func), std::forward<S>(arg1),
               std::forward<T>(arg2), std::make_index_sequence<n>{});
  }
  template <typename T, typename... Args>
  static constexpr std::array<
      T, SKIP_UNTIL_TYPE_V<SizeOfPackOfType, T,
                           std::remove_cv_t<std::remove_reference_t<Args>>...>>
  packOfType(Args&&... args) {
    decltype(packOfType<T>(std::declval<Args>()...)) array{};
    if constexpr (array.size()) {
      skipUntilType<T>(
          [&array](auto&&... skippedArgs) {
            fillArray<0>(array,
                         std::forward<decltype(skippedArgs)>(skippedArgs)...);
          },
          std::forward<Args>(args)...);
    }
    return array;
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
  static QIR_DD_Backend& getInstance(bool reinitialized = false);

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
