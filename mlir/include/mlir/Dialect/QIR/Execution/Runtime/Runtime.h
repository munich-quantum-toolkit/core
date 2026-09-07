/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Runtime.hpp
 * @brief C++ QIR runtime support.
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "mlir/Dialect/QIR/Execution/Runtime/QIR.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

/// @note this struct is purposefully not called ResultImpl to leave the Result
/// pointer opaque such that it cannot be dereferenced
struct ResultStruct {
  bool r{};
};
struct ArrayImpl {
  int32_t refcount{};
  std::vector<int8_t> data{};
  int64_t elementSize{};
};

namespace qir {

template <typename T, typename... Args>
static constexpr auto packOfType(Args&&... args) {
  constexpr size_t count =
      (size_t{0} + ... +
       static_cast<size_t>(std::is_same_v<T, std::remove_cvref_t<Args>>));
  std::array<T, count> values{};
  size_t index = 0;
  (
      [&] {
        if constexpr (std::is_same_v<T, std::remove_cvref_t<Args>>) {
          values[index++] = std::forward<Args>(args);
        }
      }(),
      ...);
  return values;
}

class Runtime {
public:
  /// The quantum state held by the runtime:
  /// - a DD package,
  /// - the root edge into that package, and
  /// - the number of qubits the state spans.
  struct QState {
    std::unique_ptr<dd::Package> dd;
    dd::vEdge edge;
    size_t numQubits;

    QState()
        : dd(std::make_unique<dd::Package>()), edge(dd::vEdge::one()),
          numQubits(0) {}

    /// Reset to a fresh empty state.
    /// If @c dd is currently populated, the existing package's `decRef` plus
    /// `garbageCollect` path is used so the package (and its internal caches)
    /// is kept warm.
    /// If @c dd was moved out (e.g., by @ref Runtime::takeState), a new package
    /// is allocated.
    auto reset() -> void {
      if (dd) {
        dd->decRef(edge);
        dd->garbageCollect();
      } else {
        dd = std::make_unique<dd::Package>();
      }
      edge = dd::vEdge::one();
      numQubits = 0;
    }
  };

  enum class OutputSchema : uint8_t { Labeled, Ordered };

private:
  friend class JitSession;

  static constexpr uintptr_t MIN_DYN_QUBIT_ADDRESS =
      dd::Package::MAX_POSSIBLE_QUBITS;
  enum class ResourceMode : uint8_t { UNKNOWN, DYNAMIC, STATIC };

  ResourceMode qubitMode;
  std::unordered_map<const Qubit*, dd::Qubit> qRegister;
  // swap gates are not executed, they are tracked here
  std::vector<dd::Qubit> qubitPermutation;
  static constexpr uintptr_t MIN_DYN_RESULT_ADDRESS = 0x10000;
  ResourceMode resultMode;
  std::unordered_map<Result*, ResultStruct> rRegister;
  std::string measurements;
  uintptr_t currentMaxQubitAddress;
  size_t currentMaxQubitId;
  uintptr_t currentMaxResultAddress;
  QState qState;
  std::mt19937_64 mt;
  std::ostream* os = &std::cout;
  // The QIR spec does not define a default output schema.
  // The runtime picks @c Labeled when a program doesn't declare one.
  OutputSchema outputSchema = OutputSchema::Labeled;
  std::vector<std::pair<std::string, std::string>> metadata;

  auto enlargeState(size_t maxQubit) -> void;
  static auto staticQubitId(const Qubit* qubit) -> dd::Qubit {
    const auto id = reinterpret_cast<uintptr_t>(qubit);
    if (id >= dd::Package::MAX_POSSIBLE_QUBITS) {
      throw std::out_of_range(
          "Static QIR qubit ID exceeds the supported qubit range");
    }
    return static_cast<dd::Qubit>(id);
  }
  static auto bind(Runtime* runtime) noexcept -> Runtime*;
  auto resolveAddress(const Qubit* qubit) -> dd::Qubit;
  auto translateAddresses(std::span<Qubit* const> qubits,
                          std::span<Qubit* const> additionalQubits = {})
      -> std::vector<dd::Qubit>;

  // Helper function to output a type (bool, int...) to @c os, honoring the
  // active @c outputSchema.
  // The label is included only in Labeled mode.
  // Tab separator between fields, newline at end.
  void outputType(const char* type, std::string_view value,
                  const char* label) const;

public:
  Runtime();
  explicit Runtime(uint64_t randomSeed);

  [[nodiscard]] static auto generateRandomSeed() -> uint64_t;
  /// Return the runtime bound to this thread. When no session is executing, a
  /// thread-local fallback keeps direct calls to the public C ABI convenient.
  static Runtime& getInstance();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  auto reset() -> void;
  auto seed(uint64_t randomSeed) -> void;
  /// Apply a row-major matrix with a runtime-sized control set.
  auto apply(std::span<const std::complex<dd::fp>> matrix,
             std::span<Qubit* const> controls, std::span<Qubit* const> targets)
      -> void;
  template <typename Matrix>
    requires requires(const Matrix& matrix) { matrix.entries(); }
  auto apply(const Matrix& matrix, std::span<Qubit* const> controls,
             std::span<Qubit* const> targets) -> void {
    apply(matrix.entries(), controls, targets);
  }
  auto applyGlobalPhase(dd::fp phase) -> void;
  template <typename... Args> auto measure(Args... args) -> void {
    const auto qubits = packOfType<Qubit*>(args...);
    const auto results = packOfType<Result*>(args...);
    static_assert(
        qubits.size() == results.size(),
        "Number of qubits and results must match. First, all qubits followed "
        "then by all results.");
    static_assert(
        qubits.size() + results.size() == sizeof...(Args),
        "Number of qubits and results must match the number of arguments. "
        "First, all qubits followed then by all results.");
    auto targets = translateAddresses(qubits);
    for (size_t i = 0; i < targets.size(); ++i) {
      targets[i] = qubitPermutation[targets[i]];
      const auto result =
          qState.dd->measureOneCollapsing(qState.edge, targets[i], mt);
      deref(results[i]).r = result == '1';
    }
  }
  auto reset(std::span<Qubit* const> qubits) -> void;
  auto swap(Qubit* qubit1, Qubit* qubit2) -> void;
  auto qAlloc() -> Qubit*;
  auto qFree(Qubit* qubit) -> void;
  auto rAlloc() -> Result*;
  auto deref(Result* result) -> ResultStruct&;
  auto rFree(Result* result) -> void;

  /// Append a measurement bit to the measurement string.
  auto appendMeasurementBit(bool result) -> void;

  /// @returns the accumulated measurement string.
  auto getMeasurements() const -> const std::string&;

  /// Move the quantum state out of the runtime.
  /// Then reset the runtime to a clean state ready for the next job.
  /// Intended for use after a @c JitSession constructed with
  /// @c Execution::StateExtraction has finished running.
  /// @returns the moved @c QState from the runtime.
  auto takeState() -> QState;

  auto setOstream(std::ostream& other) -> void;
  auto resetOstream() -> void;

  /// Emit `OUTPUT\tRESULT\t<0|1>[\tlabel]\n` to the output stream.
  auto outputResult(bool value, const char* label) const -> void;

  /// Emit `OUTPUT\tRESULT_ARRAY\t<bits>[\tlabel]\n` in memory order.
  auto outputResultArray(std::string_view values, const char* label) const
      -> void;

  /// Emit `OUTPUT\tBOOL\t<true|false>[\tlabel]\n` to the output stream.
  auto outputBool(bool value, const char* label) const -> void;

  /// Emit `OUTPUT\tINT\t<value>[\tlabel]\n` to the output stream.
  auto outputInt(int64_t value, const char* label) const -> void;

  /// Emit `OUTPUT\tDOUBLE\t<value>[\tlabel]\n` to the output stream.
  auto outputFloat(double value, const char* label) const -> void;

  /// Emit `OUTPUT\tTUPLE\t<elementCount>[\tlabel]\n` to the output stream.
  auto outputTuple(int64_t elementCount, const char* label) const -> void;

  /// Emit `OUTPUT\tARRAY\t<elementCount>[\tlabel]\n` to the output stream.
  auto outputArray(int64_t elementCount, const char* label) const -> void;

  /// Emit the HEADER records (once per submitted program):
  /// `HEADER\tschema_id\t<labeled|ordered>`
  /// `HEADER\tschema_version\t2.1`
  auto outputProgramHeader() const -> void;

  /// Emit `START\n` followed by
  /// `METADATA\toutput_labeling_schema\t<labeled|ordered>\n` (one per shot).
  auto outputShotStart() const -> void;

  /// Emit `END\t<exitCode>\n` (one per shot).
  auto outputShotEnd(int64_t exitCode = 0) const -> void;

  [[nodiscard]] auto getOutputSchema() const -> OutputSchema;
  auto setOutputSchema(OutputSchema schema) -> void;
  auto setMetadata(
      std::vector<std::pair<std::string, std::string>> entryPointMetadata)
      -> void;
};

/// Write the schema's spec-mandated literal (`labeled` or `ordered`).
auto operator<<(std::ostream& os, Runtime::OutputSchema schema)
    -> std::ostream&;

} // namespace qir
