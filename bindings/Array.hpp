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

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

#include <memory>
#include <utility>
#include <vector>

namespace mqt::bindings {
namespace nb = nanobind;

/// Transfer a native vector into a one-dimensional NumPy array.
template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>
ownedArray(std::vector<T>&& values) {
  auto storage = std::make_unique<std::vector<T>>(std::move(values));
  auto* const data = storage->data();
  const auto size = storage->size();
  const nb::capsule owner(storage.get(), [](void* ptr) noexcept {
    delete static_cast<std::vector<T>*>(ptr);
  });
  [[maybe_unused]] const auto* const released = storage.release();
  return {data, {size}, owner};
}

} // namespace mqt::bindings
