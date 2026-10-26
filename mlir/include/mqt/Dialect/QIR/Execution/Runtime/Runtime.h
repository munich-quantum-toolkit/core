/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Runtime.hpp
/// C++ QIR runtime support.

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "mqt/Dialect/QIR/Execution/Runtime/QIR.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/// Stores a measurement result behind an opaque QIR Result handle.
///
/// @note this struct is purposefully not called ResultImpl to leave the Result
/// pointer opaque such that it cannot be dereferenced
struct ResultStruct {
  bool r{};
  std::optional<dd::Qubit> deferredWire;
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
  /// Holds the runtime's quantum state.
  ///
  /// The state consists of:
  /// - a DD package,
  /// - the root edge into that package, and
  /// - the number of qubits the state spans.
  struct QState {
    std::unique_ptr<dd::Package> dd;
    dd::vEdge edge;
    size_t numQubits;

    QState()
        : dd(llvm::cantFail(dd::Package::create(0))), edge(dd::vEdge::one()),
          numQubits(0) {}

    /// Reset to a fresh empty state.
    ///
    /// If @c dd is currently populated, the existing package's `decRef` plus
    /// `garbageCollect` path is used so the package (and its internal caches)
    /// is kept warm. A moved-out package is recreated on the next quantum
    /// operation.
    void reset() {
      if (dd) {
        dd->decRef(edge);
        dd->garbageCollect();
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
  std::vector<dd::Qubit> freeQubits_;
  // swap gates are not executed, they are tracked here
  std::vector<dd::Qubit> qubitPermutation;
  static constexpr uintptr_t MIN_DYN_RESULT_ADDRESS = 0x10000;
  ResourceMode resultMode;
  std::unordered_map<Result*, ResultStruct> rRegister;
  std::optional<size_t> staticQubits_;
  std::optional<size_t> staticResults_;
  std::vector<ResultStruct> resultValues_;
  bool deferMeasurements_ = false;
  bool extractState_ = false;
  bool invalidStateExtraction_ = false;
  std::unordered_set<dd::Qubit> measuredQubits_;
  std::string measurements;
  uintptr_t currentMaxQubitAddress;
  size_t currentMaxQubitId;
  uintptr_t currentMaxResultAddress;
  QState qState;
  std::mt19937_64 mt;
  struct SamplingBranches {
    std::vector<bool> path;
    std::vector<std::pair<std::vector<bool>, size_t>> pending;
    size_t count;
    size_t position = 0;
  };
  SamplingBranches* samplingBranches_ = nullptr;
  std::vector<dd::SamplingOutput>* deferredOutputs_ = nullptr;
  llvm::raw_ostream* os = &llvm::outs();
  // Set only while building a terminal-sampling template in a
  // raw_string_ostream.
  std::vector<uint64_t>* outputResultPositions_ = nullptr;
  // The QIR spec does not define a default output schema.
  // The runtime picks @c Labeled when a program doesn't declare one.
  OutputSchema outputSchema = OutputSchema::Labeled;
  std::vector<std::pair<std::string, std::string>> metadata;

  llvm::Error enlargeState(size_t maxQubit);
  llvm::Expected<bool> measureQubit(dd::Qubit target);
  llvm::Error configureStaticResources(std::optional<size_t> qubits,
                                       std::optional<size_t> results);
  llvm::Error
  sampleMeasurements(std::span<const uintptr_t> outputs, size_t shots,
                     llvm::function_ref<void(std::string&&)> emitOwned,
                     llvm::function_ref<void(std::string_view)> emitBorrowed);
  static llvm::Expected<dd::Qubit> staticQubitId(const Qubit* qubit) {
    const auto id = reinterpret_cast<uintptr_t>(qubit);
    if (id >= dd::Package::MAX_POSSIBLE_QUBITS) {
      return llvm::createStringError(
          std::errc::result_out_of_range,
          "Static QIR qubit ID exceeds the supported qubit range");
    }
    return static_cast<dd::Qubit>(id);
  }
  static Runtime* bind(Runtime* runtime) noexcept;
  llvm::Expected<dd::Qubit> resolveAddress(const Qubit* qubit);
  llvm::Expected<llvm::SmallVector<dd::Qubit>>
  translateAddresses(std::span<Qubit* const> qubits,
                     std::span<Qubit* const> additionalQubits = {});

  // Helper function to output a type (bool, int...) to @c os, honoring the
  // active @c outputSchema.
  // The label is included only in Labeled mode.
  // Tab separator between fields, newline at end.
  void outputType(const char* type, std::string_view value,
                  const char* label) const;

public:
  Runtime();
  explicit Runtime(uint64_t randomSeed);

  [[nodiscard]] static uint64_t generateRandomSeed();
  /// Return the runtime bound to this thread.
  ///
  /// When no session is executing, a thread-local fallback keeps direct calls
  /// to the public C ABI convenient.
  static Runtime& getInstance();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  void reset();
  void seed(uint64_t randomSeed);
  /// Apply a row-major matrix with a runtime-sized control set.
  llvm::Error apply(std::span<const std::complex<dd::fp>> matrix,
                    std::span<Qubit* const> controls,
                    std::span<Qubit* const> targets);
  template <typename Matrix>
    requires requires(const Matrix& matrix) { matrix.entries(); }
  llvm::Error apply(const Matrix& matrix, std::span<Qubit* const> controls,
                    std::span<Qubit* const> targets) {
    return apply(matrix.entries(), controls, targets);
  }
  void applyGlobalPhase(dd::fp phase);
  llvm::Error measure(Qubit* qubit, Result* result);
  template <typename... Args> llvm::Error measure(Args... args) {
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
    for (size_t i = 0; i < qubits.size(); ++i) {
      if (auto error = measure(qubits[i], results[i])) {
        return error;
      }
    }
    return llvm::Error::success();
  }
  llvm::Error reset(std::span<Qubit* const> qubits);
  llvm::Error swap(Qubit* qubit1, Qubit* qubit2);
  llvm::Expected<Qubit*> qAlloc();
  llvm::Error qFree(Qubit* qubit);
  llvm::Expected<Result*> rAlloc();
  llvm::Expected<ResultStruct*> deref(Result* result);
  llvm::Error rFree(Result* result);

  /// Append a measurement bit to the measurement string.
  void
  appendMeasurementBit(bool result,
                       std::optional<dd::Qubit> deferredWire = std::nullopt);

  /// Returns the accumulated measurement string.
  const std::string& getMeasurements() const;

  /// Move the quantum state out of the runtime.
  ///
  /// Then reset the runtime to a clean state ready for the next job. Intended
  /// for use after state extraction or successful terminal sampling that
  /// reported an available uncollapsed state.
  ///
  /// @returns the moved @c QState from the runtime.
  llvm::Expected<QState> takeState();

  /// Borrow a stream that outlives execution and session teardown.
  void setOstream(llvm::raw_ostream& other);
  void resetOstream();
  /// Disable textual records while retaining measurement bits.
  void disableOutput();
  [[nodiscard]] bool hasOutput() const {
    return os != nullptr && !extractState_;
  }

  /// Emit `OUTPUT\tRESULT\t<0|1>[\tlabel]\n` to the output stream.
  void outputResult(bool value, const char* label) const;

  /// Emit `OUTPUT\tRESULT_ARRAY\t<bits>[\tlabel]\n` in memory order.
  void outputResultArray(std::string_view values, const char* label) const;

  /// Emit `OUTPUT\tBOOL\t<true|false>[\tlabel]\n` to the output stream.
  void outputBool(bool value, const char* label) const;

  /// Emit `OUTPUT\tINT\t<value>[\tlabel]\n` to the output stream.
  void outputInt(int64_t value, const char* label) const;

  /// Emit `OUTPUT\tDOUBLE\t<value>[\tlabel]\n` to the output stream.
  void outputFloat(double value, const char* label) const;

  /// Emit `OUTPUT\tTUPLE\t<elementCount>[\tlabel]\n` to the output stream.
  void outputTuple(int64_t elementCount, const char* label) const;

  /// Emit `OUTPUT\tARRAY\t<elementCount>[\tlabel]\n` to the output stream.
  void outputArray(int64_t elementCount, const char* label) const;

  /// Emit the HEADER records (once per submitted program):
  /// `HEADER\tschema_id\t<labeled|ordered>`
  /// `HEADER\tschema_version\t2.1`
  void outputProgramHeader() const;

  /// Emit `START\n` followed by
  /// `METADATA\toutput_labeling_schema\t<labeled|ordered>\n` (one per shot).
  void outputShotStart() const;

  /// Emit `END\t<exitCode>\n` (one per shot).
  void outputShotEnd(int64_t exitCode = 0) const;

  [[nodiscard]] OutputSchema getOutputSchema() const;
  void setOutputSchema(OutputSchema schema);
  void setMetadata(
      std::vector<std::pair<std::string, std::string>> entryPointMetadata);
};

/// Write the schema's spec-mandated literal (`labeled` or `ordered`).
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              Runtime::OutputSchema schema);

} // namespace qir
