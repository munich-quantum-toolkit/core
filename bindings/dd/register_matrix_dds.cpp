/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"

#include "register_dd_export.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/complex.h" // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"  // NOLINT(misc-include-cleaner)
#include "nanobind/stl/vector.h"  // NOLINT(misc-include-cleaner)

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using Matrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
Matrix getMatrix(const dd::mEdge& m, const size_t numQubits,
                 const dd::fp threshold) {
  if (numQubits > 20U) {
    throw nb::value_error("num_qubits exceeds practical limit of 20");
  }

  if (numQubits == 0U) {
    auto dataPtr = std::make_unique<std::complex<dd::fp>>(m.w);
    auto* const data = dataPtr.get();
    const nb::capsule owner(data, [](void* ptr) noexcept {
      delete static_cast<std::complex<dd::fp>*>(ptr);
    });
    [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
    return Matrix(data, {1, 1}, owner);
  }

  const auto dim = 1ULL << numQubits;
  auto dataPtr = std::make_unique<dd::CVec>(dim * dim);
  m.traverseMatrix(
      std::complex<dd::fp>{1., 0.}, 0ULL, 0ULL,
      [&dataPtr, dim](const std::size_t i, const std::size_t j,
                      const std::complex<dd::fp>& c) {
        (*dataPtr)[(i * dim) + j] = c;
      },
      numQubits, threshold);
  auto* const data = dataPtr->data();
  const nb::capsule owner(dataPtr.get(), [](void* ptr) noexcept {
    delete static_cast<dd::CVec*>(ptr);
  });
  [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
  return Matrix(data, {dim, dim}, owner);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerMatrixDDs(const nb::module_& m) {
  auto mat = nb::class_<dd::mEdge>(
      m, "MatrixDD", "A class representing a matrix decision diagram (DD).");

  mat.def("is_terminal", &dd::mEdge::isTerminal,
          "Check if the DD is a terminal node.");
  mat.def("is_zero_terminal", &dd::mEdge::isZeroTerminal,
          "Check if the DD is a zero terminal node.");

  mat.def("is_identity", &dd::mEdge::isIdentity, "up_to_global_phase"_a = true,
          R"pb(Check if the DD represents the identity matrix.

Args:
    up_to_global_phase: Whether to ignore global phase.

Returns:
    Whether the DD represents the identity matrix.)pb");

  mat.def("size", nb::overload_cast<>(&dd::mEdge::size, nb::const_),
          "Get the size of the DD by traversing it once.");

  mat.def("get_entry",
          nb::overload_cast<size_t, size_t, size_t>(&dd::mEdge::getValueByIndex,
                                                    nb::const_),
          "num_qubits"_a, "row"_a, "col"_a,
          "Get the entry of the matrix by row and column index.");

  mat.def("get_entry_by_path", &dd::mEdge::getValueByPath, "num_qubits"_a,
          "decisions"_a, R"pb(Get the entry of the matrix by decisions.

Args:
    num_qubits: The number of qubits.
    decisions: The decisions as a string of `0`, `1`, `2`, or `3`, where `decisions[i]` corresponds to the successor to follow at level `i` of the DD.
        Must be at least `num_qubits` long.

Returns:
    The entry of the matrix.)pb");

  mat.def("get_matrix", &getMatrix, "num_qubits"_a, "threshold"_a = 0.,
          R"pb(Get the matrix represented by the DD.

Args:
    num_qubits: The number of qubits.
    threshold: The threshold for not including entries in the matrix. Defaults to 0.0.

Returns:
    The matrix.

Raises:
    MemoryError: If the memory allocation fails.)pb");

  registerDDExport(mat);
}
} // namespace mqt
