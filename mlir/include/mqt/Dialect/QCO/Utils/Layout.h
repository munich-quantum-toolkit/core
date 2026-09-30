/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

namespace mlir::qco {

/// A qubit layout that maps program qubit indices to hardware qubit indices
/// without storing Values.
///
/// Program and hardware qubit indices form a dense range, respectively
/// `[0, nProgramQubits)` and `[0, nHardwareQubits)`, with `nProgramQubits <=
/// nHardwareQubits`, and every program qubit is mapped to a distinct hardware
/// qubit. Unmapped hardware slots carry a sentinel value.
/// The site count must not exceed `std::numeric_limits<T>::max()` so that
/// the sentinel cannot also denote a qubit.

/// Note that we use the terminology "hardware" and "program" qubits here,
/// because "virtual" (opposed to physical) and "static" (opposed to dynamic)
/// are C++ keywords.
template <class T>
  requires(std::is_integral_v<T> && std::is_unsigned_v<T> &&
           !std::is_same_v<T, bool>)
class Layout {
  /// Sentinel stored in `programToHardware_` and `hardwareToProgram_` entries
  /// that do not currently hold a valid index.
  constexpr static T UNMAPPED = std::numeric_limits<T>::max();

public:
  /// Construct an empty layout.
  Layout() = default;

  /// Construct and return an identity layout that maps the i-th program qubit
  /// index in `[0, nqubits)` to the i-th hardware index in `[0, nqubits)`.
  /// Sets both `nProgramQubits` and `nHardwareQubits` to `nqubits`.
  static Layout<T> identity(size_t nqubits) {
    checkSize(nqubits);
    return fromMapping(to_vector(llvm::seq<T>(static_cast<T>(nqubits))));
  }

  /// Construct and return a random layout that maps every program qubit
  /// index in `[0, nProgramQubits)` to a distinct hardware index drawn from
  /// `[0, nHardwareQubits)`.
  static Layout<T> random(size_t nProgramQubits, size_t nHardwareQubits,
                          size_t seed) {
    assert(nProgramQubits <= nHardwareQubits &&
           "cannot map more program qubits than hardware qubits");
    checkSize(nHardwareQubits);
    auto hwIndices =
        llvm::to_vector(llvm::seq<T>(static_cast<T>(nHardwareQubits)));
    llvm::shuffle(hwIndices.begin(), hwIndices.end(), std::mt19937_64{seed});

    Layout<T> layout(nProgramQubits, nHardwareQubits);
    for (const auto prog : llvm::seq(static_cast<T>(nProgramQubits))) {
      layout.add(prog, hwIndices[prog]);
    }
    return layout;
  }

  /// Construct a layout from a bijective program-to-hardware mapping,
  /// where mapping[prog] = hw.
  /// Sets both `nProgramQubits` and `nHardwareQubits` to `mapping.size()`.
  static Layout<T> fromMapping(ArrayRef<T> mapping) {
    checkSize(mapping.size());
    llvm::SmallBitVector seen(mapping.size());
    for (const auto hw : mapping) {
      if (hw >= mapping.size() || seen.test(hw)) {
        llvm::reportFatalUsageError("mapping must be a permutation");
      }
      seen.set(hw);
    }

    Layout<T> layout(mapping.size(), mapping.size());
    for (const auto [prog, hw] : enumerate(mapping)) {
      layout.add(static_cast<T>(prog), hw);
    }
    return layout;
  }

  /// Insert a program:hardware index mapping.
  /// Requires `prog < nProgramQubits`, `hw < nHardwareQubits`, and that
  /// neither `prog` nor `hw` has been mapped previously.
  void add(T prog, T hw) {
    assert(prog < programToHardware_.size() && "program index out of bounds");
    assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
    assert(programToHardware_[prog] == UNMAPPED &&
           "program index already mapped");
    assert(hardwareToProgram_[hw] == UNMAPPED &&
           "hardware index already mapped");
    programToHardware_[prog] = hw;
    hardwareToProgram_[hw] = prog;
  }

  /// Lookup and return program index for a hardware index.
  [[nodiscard]] T getProgramIndex(T hw) const {
    assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
    const auto prog = hardwareToProgram_[hw];
    assert(prog != UNMAPPED && "hardware index not mapped");
    return prog;
  }

  /// Lookup and return hardware index for a program index.
  [[nodiscard]] T getHardwareIndex(T prog) const {
    assert(prog < programToHardware_.size() && "program index out of bounds");
    const auto hw = programToHardware_[prog];
    assert(hw != UNMAPPED && "program index not mapped");
    return hw;
  }

  /// Lookup and return multiple hardware indices at once.
  template <typename... ProgIndices>
    requires(sizeof...(ProgIndices) > 0) &&
            ((std::is_convertible_v<ProgIndices, T>) && ...)
  [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
    return std::tuple{getHardwareIndex(static_cast<T>(progs))...};
  }

  /// Lookup and return multiple program indices at once.
  template <typename... HwIndices>
    requires(sizeof...(HwIndices) > 0) &&
            ((std::is_convertible_v<HwIndices, T>) && ...)
  [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
    return std::tuple{getProgramIndex(static_cast<T>(hws))...};
  }

  /// Return true if `hw` currently has a program qubit assigned to it.
  [[nodiscard]] bool hasProgramAt(T hw) const {
    assert(hw < hardwareToProgram_.size() && "hardware index out of bounds");
    return hardwareToProgram_[hw] != UNMAPPED;
  }

  /// Swap the mapping to program indices of two hardware indices.
  /// Both sides must currently have a program qubit assigned.
  void swap(T hwA, T hwB) {
    assert(hwA < hardwareToProgram_.size() && "hardware index out of bounds");
    assert(hwB < hardwareToProgram_.size() && "hardware index out of bounds");
    if (hwA == hwB) {
      return;
    }
    const auto progA = hardwareToProgram_[hwA];
    const auto progB = hardwareToProgram_[hwB];
    assert(progA != UNMAPPED && "hardware index not mapped");
    assert(progB != UNMAPPED && "hardware index not mapped");
    hardwareToProgram_[hwA] = progB;
    hardwareToProgram_[hwB] = progA;
    programToHardware_[progA] = hwB;
    programToHardware_[progB] = hwA;
  }

  /// Return the number of program qubits this layout was declared with.
  [[nodiscard]] size_t nProgramQubits() const {
    return programToHardware_.size();
  }

  /// Return the number of hardware qubits this layout was declared with.
  [[nodiscard]] size_t nHardwareQubits() const {
    return hardwareToProgram_.size();
  }

  /// Return a view of the program to hardware mapping of length
  /// `nProgramQubits()`, where entry `prog` is the hardware index assigned to
  /// program qubit `prog`. Requires every program qubit to be mapped.
  [[nodiscard]] ArrayRef<T> getProgramToHardware() const {
#ifndef NDEBUG
    for (const auto hw : programToHardware_) {
      assert(hw != UNMAPPED && "program qubit not mapped");
    }
#endif
    return programToHardware_;
  }

  /// Compare two layouts for equality.
  [[nodiscard]] bool operator==(const Layout& other) const {
    return programToHardware_ == other.programToHardware_ &&
           hardwareToProgram_ == other.hardwareToProgram_;
  }

private:
  static void checkSize(size_t nqubits) {
    if (nqubits > UNMAPPED) {
      llvm::reportFatalUsageError("layout exceeds qubit index capacity");
    }
  }

  Layout(size_t nProgramQubits, size_t nHardwareQubits)
      : programToHardware_(nProgramQubits, UNMAPPED),
        hardwareToProgram_(nHardwareQubits, UNMAPPED) {}

  /// Maps a program qubit index to its hardware index.
  /// Size equals `nProgramQubits()`.
  SmallVector<T> programToHardware_;
  /// Maps a hardware qubit index to its program index.
  /// Size equals `nHardwareQubits()`.
  SmallVector<T> hardwareToProgram_;
};
} // namespace mlir::qco
