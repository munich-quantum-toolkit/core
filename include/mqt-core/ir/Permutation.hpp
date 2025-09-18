/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "operations/Control.hpp"

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <map>
#include <utility>

namespace qc {
class Permutation {
  std::map<Qubit, Qubit> permutation;

public:
  [[nodiscard]] auto apply(const Controls& controls) const -> Controls;
  [[nodiscard]] auto apply(const Targets& targets) const -> Targets;
  [[nodiscard]] auto apply(Qubit qubit) const -> Qubit;
  [[nodiscard]] auto maxKey() const -> Qubit;
  [[nodiscard]] auto maxValue() const -> Qubit;

  /// Constructors
  Permutation() = default;
  template <class InputIt>
  Permutation(InputIt first, InputIt last) : permutation(first, last) {}
  Permutation(const std::initializer_list<std::pair<const Qubit, Qubit>> init)
      : permutation(init) {}

  /// Returns an iterator to the beginning
  [[nodiscard]] auto begin() noexcept -> decltype(permutation)::iterator {
    return permutation.begin();
  }
  [[nodiscard]] auto begin() const noexcept
      -> decltype(permutation)::const_iterator {
    return permutation.begin();
  }
  [[nodiscard]] auto cbegin() const noexcept -> auto {
    return permutation.cbegin();
  }

  /// Returns an iterator to the end
  [[nodiscard]] auto end() noexcept -> decltype(permutation)::iterator {
    return permutation.end();
  }
  [[nodiscard]] auto end() const noexcept
      -> decltype(permutation)::const_iterator {
    return permutation.end();
  }
  [[nodiscard]] auto cend() const noexcept -> auto {
    return permutation.cend();
  }

  /// Returns a reverse iterator to the beginning
  [[nodiscard]] auto rbegin() noexcept
      -> decltype(permutation)::reverse_iterator {
    return permutation.rbegin();
  }
  [[nodiscard]] auto rbegin() const noexcept
      -> decltype(permutation)::const_reverse_iterator {
    return permutation.rbegin();
  }
  [[nodiscard]] auto crbegin() const noexcept -> auto {
    return permutation.crbegin();
  }

  /// Returns a reverse iterator to the end
  [[nodiscard]] auto rend() noexcept
      -> decltype(permutation)::reverse_iterator {
    return permutation.rend();
  }
  [[nodiscard]] auto rend() const noexcept
      -> decltype(permutation)::const_reverse_iterator {
    return permutation.rend();
  }
  [[nodiscard]] auto crend() const noexcept -> auto {
    return permutation.crend();
  }

  /// Checks whether the permutation is empty
  [[nodiscard]] auto empty() const -> bool { return permutation.empty(); }

  /// Returns the number of elements
  [[nodiscard]] auto size() const -> std::size_t { return permutation.size(); }

  /// Clears the permutation
  void clear() { permutation.clear(); }

  /// Finds element with specific key
  [[nodiscard]] auto find(const Qubit qubit)
      -> decltype(permutation.find(qubit)) {
    return permutation.find(qubit);
  }
  [[nodiscard]] auto find(const Qubit qubit) const
      -> decltype(permutation.find(qubit)) {
    return permutation.find(qubit);
  }

  /// Returns the number of elements with specific key
  [[nodiscard]] auto count(const Qubit qubit) const -> std::size_t {
    return permutation.count(qubit);
  }

  /// Access specified element with bounds checking
  [[nodiscard]] auto at(const Qubit qubit) const -> Qubit {
    return permutation.at(qubit);
  }

  /// Access specified element with bounds checking
  [[nodiscard]] auto at(const Qubit qubit) -> Qubit& {
    return permutation.at(qubit);
  }

  /// Access or insert specified element
  [[nodiscard]] auto operator[](const Qubit qubit) -> Qubit& {
    return permutation[qubit];
  }

  /// Inserts elements or nodes
  auto insert(const std::pair<const Qubit, Qubit>& value) -> auto {
    return permutation.insert(value);
  }
  template <class InputIt> auto insert(InputIt first, InputIt last) -> void {
    permutation.insert(first, last);
  }
  auto insert(const std::initializer_list<std::pair<const Qubit, Qubit>> init)
      -> void {
    permutation.insert(init);
  }

  /// Constructs element in-place
  template <class... Args> auto emplace(Args&&... args) -> auto {
    return permutation.emplace(std::forward<Args>(args)...);
  }

  // NOLINTBEGIN(readability-identifier-naming)

  /// Inserts in-place if the key does not exist, does nothing otherwise
  template <class... Args>
  auto try_emplace(const Qubit key, Args&&... args) -> auto {
    return permutation.try_emplace(key, std::forward<Args>(args)...);
  }

  /// Inserts an element or assigns to the current element if the key already
  /// exists
  auto insert_or_assign(const Qubit key, const Qubit value) -> auto {
    return permutation.insert_or_assign(key, value);
  }

  // NOLINTEND(readability-identifier-naming)

  /// Erases elements
  auto erase(const Qubit qubit) -> std::size_t {
    return permutation.erase(qubit);
  }
  auto erase(const decltype(permutation)::const_iterator pos)
      -> decltype(permutation)::iterator {
    return permutation.erase(pos);
  }
  auto erase(const decltype(permutation)::const_iterator first,
             const decltype(permutation)::const_iterator last)
      -> decltype(permutation)::iterator {
    return permutation.erase(first, last);
  }

  /// Swaps the contents
  void swap(Permutation& other) noexcept {
    permutation.swap(other.permutation);
  }

  /// Lexicographically compares the values in the map
  [[nodiscard]] auto operator<(const Permutation& other) const -> bool {
    return permutation < other.permutation;
  }
  [[nodiscard]] auto operator<=(const Permutation& other) const -> bool {
    return permutation <= other.permutation;
  }
  [[nodiscard]] auto operator>(const Permutation& other) const -> bool {
    return permutation > other.permutation;
  }
  [[nodiscard]] auto operator>=(const Permutation& other) const -> bool {
    return permutation >= other.permutation;
  }
  [[nodiscard]] auto operator==(const Permutation& other) const -> bool {
    return permutation == other.permutation;
  }
  [[nodiscard]] auto operator!=(const Permutation& other) const -> bool {
    return permutation != other.permutation;
  }
};
} // namespace qc

// define hash function for Permutation
template <> struct std::hash<qc::Permutation> {
  std::size_t operator()(const qc::Permutation& p) const noexcept {
    std::size_t seed = 0;
    for (const auto& [k, v] : p) {
      qc::hashCombine(seed, k);
      qc::hashCombine(seed, v);
    }
    return seed;
  }
};
