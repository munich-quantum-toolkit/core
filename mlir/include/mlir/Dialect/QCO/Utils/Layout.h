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
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
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

/// Note that we use the terminology "hardware" and "program" qubits here,
/// because "virtual" (opposed to physical) and "static" (opposed to dynamic)
/// are C++ keywords.
class Layout {
public:
  /// Construct an empty layout.
  Layout() = default;

  /// Construct and return a random layout that maps every program qubit
  /// index in `[0, nProgramQubits)` to a distinct hardware index drawn from
  /// `[0, nHardwareQubits)`.
  static Layout random(size_t nProgramQubits, size_t nHardwareQubits,
                       size_t seed);

  /// Construct a layout from a bijective program-to-hardware mapping,
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
  /// Both sides must currently have a program qubit assigned.
  void swap(size_t hwA, size_t hwB);

  /// Return the number of program qubits this layout was declared with.
  [[nodiscard]] size_t nProgramQubits() const;

  /// Return the number of hardware qubits this layout was declared with.
  [[nodiscard]] size_t nHardwareQubits() const;

  /// Return a view of the program to hardware mapping of length
  /// `nProgramQubits()`, where entry `prog` is the hardware index assigned to
  /// program qubit `prog`. Requires every program qubit to be mapped.
  [[nodiscard]] ArrayRef<size_t> getProgramToHardware() const;

  /// Compare two layouts for equality.
  [[nodiscard]] bool operator==(const Layout& other) const {
    return programToHardware_ == other.programToHardware_ &&
           hardwareToProgram_ == other.hardwareToProgram_;
  }

private:
  Layout(size_t nProgramQubits, size_t nHardwareQubits);

  /// Maps a program qubit index to its hardware index.
  /// Size equals `nProgramQubits()`.
  SmallVector<size_t> programToHardware_;
  /// Maps a hardware qubit index to its program index.
  /// Size equals `nHardwareQubits()`.
  SmallVector<size_t> hardwareToProgram_;
};
} // namespace mlir::qco
