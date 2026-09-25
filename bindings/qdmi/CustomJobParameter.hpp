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

#include "qdmi/common/Common.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // NOLINT(misc-include-cleaner)
#include "nanobind/stl/variant.h" // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace nanobind::detail {

/// Keep bytes distinct from strings, whose QDMI payload includes a terminator.
template <> struct type_caster<qdmi::CustomJobParameter> {
  NB_TYPE_CASTER(qdmi::CustomJobParameter,
                 const_name("str | bool | int | float | bytes"))

  // nanobind requires these caster method names.
  // NOLINTNEXTLINE(readability-identifier-naming)
  bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) {
    if (isinstance<bytes>(src)) {
      const auto data = borrow<bytes>(src);
      const auto buffer =
          std::span{static_cast<const std::byte*>(data.data()), data.size()};
      value = std::vector<std::byte>(buffer.begin(), buffer.end());
      return true;
    }
    using Scalar = std::variant<std::string, bool, int, double>;
    make_caster<Scalar> scalar;
    if (!scalar.from_python(src, flags, cleanup)) {
      return false;
    }
    value = std::visit(
        [](const auto& item) -> qdmi::CustomJobParameter { return item; },
        static_cast<Scalar&>(scalar));
    return true;
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static handle from_cpp(const qdmi::CustomJobParameter& src, rv_policy policy,
                         cleanup_list* cleanup) {
    return std::visit(
        [&](const auto& item) -> handle {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::vector<std::byte>>) {
            return bytes(item.data(), item.size()).release();
          } else {
            return make_caster<T>::from_cpp(item, policy, cleanup);
          }
        },
        src);
  }
};

} // namespace nanobind::detail
