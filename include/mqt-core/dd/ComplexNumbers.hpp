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

#include "dd/CachedEdge.hpp"
#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/RealNumberUniqueTable.hpp"

#include <complex>
#include <cstddef>

namespace dd {

struct ComplexValue;

/// A class for managing complex numbers in the DD package.
class ComplexNumbers {

public:
  /// Default constructor.
  explicit ComplexNumbers(RealNumberUniqueTable& table)
      : uniqueTable(&table) {};
  /// Default destructor.
  ~ComplexNumbers() = default;

  /**
   * @brief Set the numerical tolerance for comparisons of floats.
   * @param tol The new tolerance.
   */
  static void setTolerance(fp tol) noexcept;

  /**
   * @brief Compute the squared magnitude of a complex number.
   * @param a The complex number.
   * @returns The squared magnitude.
   */
  [[nodiscard]] static fp mag2(const Complex& a) noexcept;

  /**
   * @brief Compute the magnitude of a complex number.
   * @param a The complex number.
   * @returns The magnitude.
   */
  [[nodiscard]] static fp mag(const Complex& a) noexcept;

  /**
   * @brief Compute the argument of a complex number.
   * @param a The complex number.
   * @returns The argument.
   */
  [[nodiscard]] static fp arg(const Complex& a) noexcept;

  /**
   * @brief Compute the complex conjugate of a complex number.
   * @param a The complex number.
   * @returns The complex conjugate.
   * @note Conjugation is efficiently handled by just flipping the sign of the
   * imaginary pointer.
   */
  [[nodiscard]] static Complex conj(const Complex& a) noexcept;

  /**
   * @brief Compute the negation of a complex number.
   * @param a The complex number.
   * @returns The negation.
   * @note Negation is efficiently handled by just flipping the sign of both
   * pointers.
   */
  [[nodiscard]] static Complex neg(const Complex& a) noexcept;

  /**
   * @brief Lookup a complex value in the complex table; if not found add it.
   * @param c The complex number.
   * @return The found or added complex number.
   */
  [[nodiscard]] Complex lookup(const Complex& c);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const std::complex<fp>& c);

  /**
   * @see lookup(fp r, fp i)
   */
  [[nodiscard]] Complex lookup(const ComplexValue& c);

  /**
   * @brief Lookup a real value in the complex table; if not found add it.
   * @param r The real number.
   * @return The found or added complex number with real part r and imaginary
   * part zero.
   */
  [[nodiscard]] Complex lookup(fp r);

  /**
   * @brief Lookup a complex value in the complex table; if not found add it.
   * @param r The real part.
   * @param i The imaginary part.
   * @return The found or added complex number.
   * @see ComplexTable::lookup
   */
  [[nodiscard]] Complex lookup(fp r, fp i);

  /**
   * @brief Turn CachedEdge into Edge via lookup.
   * @tparam Node The type of the node.
   * @param ce The cached edge.
   * @return The edge with looked-up weight. The zero terminal if the new weight
   * is exactly zero.
   */
  template <class Node>
  [[nodiscard]] Edge<Node> lookup(const CachedEdge<Node>& ce) {
    auto e = Edge<Node>{ce.p, lookup(ce.w)};
    if (e.w.exactlyZero()) {
      e.p = Node::getTerminal();
    }
    return e;
  }

  /**
   * @brief Increment the reference count of a complex number.
   * @details This is a pass-through function that increments the reference
   * count of the real and imaginary parts of the given complex number.
   * @param c The complex number
   * @see RealNumberUniqueTable::incRef(RealNumber*)
   */
  void incRef(const Complex& c) const noexcept;

  /**
   * @brief Decrement the reference count of a complex number.
   * @details This is a pass-through function that decrements the reference
   * count of the real and imaginary parts of the given complex number.
   * @param c The complex number
   * @see RealNumberUniqueTable::decRef(RealNumber*)
   */
  void decRef(const Complex& c) const noexcept;

  /**
   * @brief Check whether a complex number is one of the static ones.
   * @param c The complex number.
   * @return Whether the complex number is one of the static ones.
   */
  [[nodiscard]] static constexpr bool isStaticComplex(const Complex& c) {
    return c.exactlyZero() || c.exactlyOne();
  }

  /**
   * @brief Get the number of stored real numbers.
   * @return The number of stored real numbers.
   */
  [[nodiscard]] std::size_t realCount() const noexcept;

private:
  /// A pointer to the unique table to use for calculations
  RealNumberUniqueTable* uniqueTable;
};
} // namespace dd
