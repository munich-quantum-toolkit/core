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

#include <cstddef>

namespace mqt::test {

template <typename BuilderT> struct NamedBuilder {
  const char* name = nullptr;
  void (*fn)(BuilderT&) = nullptr;

  constexpr NamedBuilder(const char* nameIn, void (*fnIn)(BuilderT&)) noexcept
      : name(nameIn), fn(fnIn) {}

  // NOLINTNEXTLINE(*-explicit-constructor)
  constexpr NamedBuilder(std::nullptr_t) noexcept : fn(nullptr) {}

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return fn != nullptr;
  }
};

template <typename BuilderT>
[[nodiscard]] constexpr NamedBuilder<BuilderT>
namedBuilder(const char* name, void (*fn)(BuilderT&)) noexcept {
  return NamedBuilder<BuilderT>{name, fn};
}

[[nodiscard]] constexpr const char* displayName(const char* name) noexcept {
  return name != nullptr ? name : "<null>";
}

} // namespace mqt::test

#define MQT_NAMED_BUILDER(fn) ::mqt::test::namedBuilder(#fn, fn)
