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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace mlir::qco {

/// A qubit layout that maps program qubit indices to hardware qubit indices
/// without storing Values.
///
/// The mapping is a partial injection: at most `nProgramQubits` program qubits
/// are placed on distinct hardware qubits drawn from `[0, nHardwareQubits)`.
/// The two bounds may differ (usually `nProgramQubits <= nHardwareQubits`), so
/// some hardware qubits may remain unplaced.
///
/// Note that we use the terminology "hardware" and "program" qubits here,
/// because "virtual" (opposed to physical) and "static" (opposed to dynamic)
/// are C++ keywords.
class Layout {
public:
  /// Construct an empty layout.
  Layout() = default;

  /// Construct and return a random layout that places every program qubit
  /// index in `[0, nProgramQubits)` on a distinct hardware index drawn from
  /// `[0, nHardwareQubits)`.
  static Layout random(size_t nProgramQubits, size_t nHardwareQubits,
                       size_t seed);

  /// Construct a layout from a program-to-hardware mapping,
  /// where mapping[prog] = hw.
  /// Sets both `nProgramQubits` and `nHardwareQubits` to `mapping.size()`.
  static Layout fromMapping(ArrayRef<size_t> mapping);

  /// Insert a program:hardware index mapping.
  /// Requires `prog < nProgramQubits`, `hw < nHardwareQubits`, and that
  /// neither `prog` nor `hw` has been mapped previously.
  void add(size_t prog, size_t hw);

  /// Lookup and return program index for a hardware index.
  [[nodiscard]] size_t getProgramIndex(size_t hw) const;

  /// Lookup and return hardware index for a program index.
  [[nodiscard]] size_t getHardwareIndex(size_t prog) const;

  /// Lookup and return multiple hardware indices at once.
  template <typename... ProgIndices>
    requires(sizeof...(ProgIndices) > 0) &&
            ((std::is_convertible_v<ProgIndices, size_t>) && ...)
  [[nodiscard]] auto getHardwareIndices(ProgIndices... progs) const {
    return std::tuple{getHardwareIndex(static_cast<size_t>(progs))...};
  }

  /// Lookup and return multiple program indices at once.
  template <typename... HwIndices>
    requires(sizeof...(HwIndices) > 0) &&
            ((std::is_convertible_v<HwIndices, size_t>) && ...)
  [[nodiscard]] auto getProgramIndices(HwIndices... hws) const {
    return std::tuple{getProgramIndex(static_cast<size_t>(hws))...};
  }

  /// Return true if `hw` currently has a program qubit assigned to it.
  [[nodiscard]] bool hasProgramAt(size_t hw) const;

  /// Swap the mapping to program indices of two hardware indices.
  /// Either side may be currently unplaced; the empty state is treated as a
  /// legitimate value that also gets swapped.
  void swap(size_t hwA, size_t hwB);

  /// Return the number of program qubits this layout was declared with.
  [[nodiscard]] size_t nProgramQubits() const;

  /// Return the number of hardware qubits this layout was declared with.
  [[nodiscard]] size_t nHardwareQubits() const;

  /// Return the program to hardware mapping as a materialized vector of
  /// length `nProgramQubits()`, where entry `prog` is the hardware index
  /// assigned to program qubit `prog`. Requires every program qubit to be
  /// placed.
  [[nodiscard]] SmallVector<size_t> getProgramToHardware() const;

  /// Compare two layouts for equality.
  [[nodiscard]] bool operator==(const Layout& other) const {
    return programToHardware_ == other.programToHardware_;
  }

private:
  Layout(const size_t nProgramQubits, const size_t nHardwareQubits)
      : nProgramQubits_(nProgramQubits), nHardwareQubits_(nHardwareQubits) {}

  /// Maps a program qubit index to its hardware index.
  DenseMap<size_t, size_t> programToHardware_;
  /// Maps a hardware qubit index to its program index.
  DenseMap<size_t, size_t> hardwareToProgram_;
  /// Number of program qubits this layout was declared with.
  size_t nProgramQubits_ = 0;
  /// Number of hardware qubits this layout was declared with.
  size_t nHardwareQubits_ = 0;
};
} // namespace mlir::qco
