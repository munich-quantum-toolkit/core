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

#include "Result.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace mqt::bindings {
namespace nb = nanobind;

template <class T>
using Array =
    nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/// Expose `size` elements at `data` as a NumPy array that owns `storage`.
template <class T, class Storage>
Array<T> ownedArray(Storage storage, void* const data, const size_t size) {
  auto owned = std::make_unique<Storage>(std::move(storage));
  const nb::capsule owner(owned.get(), [](void* ptr) noexcept {
    delete static_cast<Storage*>(ptr);
  });
  [[maybe_unused]] const auto* const released = owned.release();
  return {data, {size}, owner};
}

/// Transfer a native vector into a one-dimensional NumPy array.
template <class T> Array<T> ownedArray(std::vector<T>&& values) {
  auto* const data = values.data();
  const auto size = values.size();
  return ownedArray<T>(std::move(values), data, size);
}

/// Read a job result of `double` or `std::complex<double>` elements into a new
/// NumPy array without initializing its storage first.
///
/// Both device calls run without the GIL.
template <class T, class Job, class Result>
Array<T> readResultArray(const Job& job, const Result result,
                         const size_t programIndex) {
  // std::complex value-initializes, so allocate its layout-compatible doubles.
  static_assert(sizeof(T) % sizeof(double) == 0);
  constexpr auto components = sizeof(T) / sizeof(double);
  const auto size = [&] {
    const nb::gil_scoped_release release;
    return unwrap(job.getResultSize(result, programIndex));
  }();
  const auto count = size / sizeof(T);
  auto storage = std::make_unique_for_overwrite<double[]>(count * components);
  auto* const data = storage.get();
  {
    const nb::gil_scoped_release release;
    unwrap(job.readResult(
        result, std::as_writable_bytes(std::span(data, count * components)),
        programIndex));
  }
  return ownedArray<T>(std::move(storage), data, count);
}

} // namespace mqt::bindings
