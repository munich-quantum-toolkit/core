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

/// Read a job result into a new one-dimensional NumPy array of `dtype`.
///
/// The device writes directly into the uninitialized array; both device calls
/// run without the GIL.
template <class T, class Job, class Result>
nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>
readResultArray(const Job& job, const Result result, const size_t programIndex,
                const char* const dtype) {
  const auto size = [&] {
    const nb::gil_scoped_release release;
    return unwrap(job.getResultSize(result, programIndex));
  }();
  auto array = nb::cast<
      nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(
      nb::module_::import_("numpy").attr("empty")(size / sizeof(T), dtype));
  {
    const nb::gil_scoped_release release;
    unwrap(job.readResult(
        result, std::as_writable_bytes(std::span(array.data(), array.size())),
        programIndex));
  }
  return array;
}

} // namespace mqt::bindings
