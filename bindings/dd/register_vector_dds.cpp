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
#include "dd/Node.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/complex.h" // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string.h"  // NOLINT(misc-include-cleaner)
#include "nanobind/stl/vector.h"  // NOLINT(misc-include-cleaner)
#include "register_dd_export.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using Vector = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<1>>;

// NOLINTNEXTLINE(misc-use-internal-linkage)
Vector getVector(const dd::vEdge& v, const dd::fp threshold) {
  auto dataPtr = std::make_unique<dd::CVec>(v.getVector(threshold));
  auto* const data = dataPtr->data();
  const auto size = dataPtr->size();
  const nb::capsule owner(dataPtr.get(), [](void* ptr) noexcept {
    delete static_cast<dd::CVec*>(ptr);
  });
  [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
  return Vector(data, {size}, owner);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerVectorDDs(const nb::module_& m) {
  auto vec = nb::class_<dd::vEdge>(
      m, "VectorDD", "A class representing a vector decision diagram (DD).");

  vec.def("is_terminal", &dd::vEdge::isTerminal,
          "Check if the DD is a terminal node.");

  vec.def("is_zero_terminal", &dd::vEdge::isZeroTerminal,
          "Check if the DD is a zero terminal node.");

  vec.def("size", nb::overload_cast<>(&dd::vEdge::size, nb::const_),
          "Get the size of the DD by traversing it once.");

  vec.def(
      "__getitem__",
      [](const dd::vEdge& v, nb::ssize_t idx) {
        const auto numQubits =
            v.isTerminal() ? 0U : static_cast<size_t>(v.p->v) + 1U;
        auto index = static_cast<size_t>(idx);
        constexpr auto digits = std::numeric_limits<size_t>::digits;
        if (numQubits < digits) {
          const auto length = size_t{1} << numQubits;
          if (idx < 0) {
            if (size_t{0} - index > length) {
              throw nb::index_error();
            }
            index += length;
          }
          if (index >= length) {
            throw nb::index_error();
          }
        } else if (idx < 0 && numQubits > digits) {
          /// Sign-extend negative indices beyond the native index width.
          auto decisions = std::string(numQubits, '1');
          for (auto bit = 0U; bit < digits; ++bit) {
            decisions[bit] = ((index >> bit) & 1U) != 0U ? '1' : '0';
          }
          return v.getValueByPath(numQubits, decisions);
        }
        return v.getValueByIndex(index);
      },
      "key"_a, "Get the amplitude of a basis state by index.");

  vec.def(
      "get_amplitude",
      [](const dd::vEdge& v, const size_t numQubits,
         const std::string& decisions) {
        return v.getValueByPath(numQubits, decisions);
      },
      "num_qubits"_a, "decisions"_a,
      R"pb(Get the amplitude of a basis state by decisions.

Args:
    num_qubits: The number of qubits.
    decisions: The decisions as a string of bits (`0` or `1`), where `decisions[i]` corresponds to the successor to follow at level `i` of the DD.
        Must be at least `num_qubits` long.

Returns:
    The amplitude of the basis state.)pb");

  vec.def("get_vector", &getVector, "threshold"_a = 0.,
          R"pb(Get the state vector represented by the DD.

Args:
    threshold: The threshold for not including amplitudes in the state vector. Defaults to 0.0.

Returns:
    The state vector.

Raises:
    MemoryError: If the memory allocation fails.)pb");

  registerDDExport(vec);
}

} // namespace mqt
