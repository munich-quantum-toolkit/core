/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <array>
#include <cassert>
#include <vector>

namespace dd {

template <class T> class SmallSpan {
public:
  SmallSpan(T* begin, std::uint8_t size) : begin_(begin), size_(size) {}

  template <std::size_t N>
  explicit SmallSpan(std::array<T, N>& arr)
      : begin_(arr.data()), size_(static_cast<std::uint8_t>(N)) {
    assert(N <= std::numeric_limits<std::uint8_t>::max());
  }

  explicit SmallSpan(std::vector<T>& vec)
      : begin_(vec.data()), size_(static_cast<std::uint8_t>(vec.size())) {
    assert(vec.size() <= std::numeric_limits<std::uint8_t>::max());
  }

  bool operator==(const SmallSpan& other) const {
    if (size() != other.size()) {
      return false;
    }
    return std::equal(begin(), end(), other.begin());
  }

  [[nodiscard]] inline constexpr const T* begin() const noexcept {
    return begin_;
  }
  [[nodiscard]] inline constexpr const T* end() const noexcept {
    return begin_ + size_;
  }

  [[nodiscard]] inline constexpr const T* cbegin() const noexcept {
    return begin();
  }
  [[nodiscard]] inline constexpr const T* cend() const noexcept {
    return end();
  }

  [[nodiscard]] inline constexpr T* begin() noexcept { return begin_; }
  [[nodiscard]] inline constexpr T* end() noexcept { return begin_ + size_; }

  [[nodiscard]] inline constexpr std::uint8_t size() const noexcept {
    return size_;
  }

  [[nodiscard]] inline constexpr const T&
  operator[](std::size_t i) const noexcept {
    return begin_[i];
  }

  [[nodiscard]] inline constexpr T& operator[](std::size_t i) noexcept {
    return begin_[i];
  }

  [[nodiscard]] inline constexpr const T& at(std::size_t i) const {
    if (i >= size()) {
      throw std::out_of_range("Index out of range");
    }
    return begin_[i];
  }

  [[nodiscard]] inline constexpr T& at(std::size_t i) {
    if (i >= size()) {
      throw std::out_of_range("Index out of range");
    }
    return begin_[i];
  }

private:
  T* begin_;
  std::uint8_t size_;
};

} // namespace dd
