#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace mlir::qco {

/// A qubit layout that maps program and hardware indices without
/// storing Values. Used for efficient memory usage when Value tracking isn't
/// needed.
///
/// Note that we use the terminology "hardware" and "program" qubits
/// here, because "virtual" (opposed to physical) and "static" (opposed to
/// dynamic) are C++ keywords.
class Layout {
public:
  /// Construct and return a random layout with size `nqubits`.
  static Layout random(size_t nqubits, size_t seed);

  /// Insert program:hardware index mapping.
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

  /// Swap the mapping to program indices of two hardware indices.
  void swap(size_t hwA, size_t hwB);

  /// Return the number of qubits managed by the layout.
  [[nodiscard]] size_t nqubits() const;

  /// Return the program to hardware mapping.
  [[nodiscard]] ArrayRef<size_t> getProgramToHardware() const;

protected:
  /// Maps a program qubit index to its hardware index.
  SmallVector<size_t> programToHardware_;
  /// Maps a hardware qubit index to its program index.
  SmallVector<size_t> hardwareToProgram_;

private:
  explicit Layout(const size_t nqubits)
      : programToHardware_(nqubits), hardwareToProgram_(nqubits) {}
};
} // namespace mlir::qco