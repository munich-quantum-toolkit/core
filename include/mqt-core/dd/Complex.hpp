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

#include "dd/DDDefinitions.hpp"
#include "dd/RealNumber.hpp"
#include "ir/Definitions.hpp"

#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>

namespace dd {

struct RealNumber;
struct ComplexValue;

/// A complex number represented by two pointers to compute table entries.
struct Complex {
  /// Compute table entry for the real part.
  RealNumber* r;
  /// Compute table entry for the imaginary part.
  RealNumber* i;

  /**
   * @brief The static constant for the complex number zero.
   * @return A complex number with real and imaginary part equal to zero.
   */
  static constexpr Complex zero() noexcept {
    return {&constants::zero, &constants::zero};
  }

  /**
   * @brief The static constant for the complex number one.
   * @return A complex number with real part equal to one and imaginary part
   * equal to zero.
   */
  static constexpr Complex one() noexcept {
    return {&constants::one, &constants::zero};
  }

  /**
   * @brief Check whether the complex number is exactly equal to zero.
   * @returns True if the complex number is exactly equal to zero, false
   * otherwise.
   * @see RealNumber::exactlyZero
   */
  [[nodiscard]] bool exactlyZero() const noexcept {
    return RealNumber::exactlyZero(r) && RealNumber::exactlyZero(i);
  }

  /**
   * @brief Check whether the complex number is exactly equal to one.
   * @returns True if the complex number is exactly equal to one, false
   * otherwise.
   * @see RealNumber::exactlyOne
   * @see RealNumber::exactlyZero
   */
  [[nodiscard]] bool exactlyOne() const noexcept {
    return RealNumber::exactlyOne(r) && RealNumber::exactlyZero(i);
  }

  /**
   * @brief Check whether the complex number is approximately equal to the
   * given complex number.
   * @param c The complex number to compare to.
   * @returns True if the complex number is approximately equal to the given
   * complex number, false otherwise.
   * @see RealNumber::approximatelyEquals
   */
  [[nodiscard]] bool approximatelyEquals(const Complex& c) const noexcept;

  /**
   * @brief Check whether the complex number is approximately equal to zero.
   * @returns True if the complex number is approximately equal to zero, false
   * otherwise.
   * @see RealNumber::approximatelyZero
   */
  [[nodiscard]] bool approximatelyZero() const noexcept;

  /**
   * @brief Convert the complex number to a string.
   * @param formatted Whether to apply special formatting to the numbers.
   * @param precision The precision to use for the numbers.
   * @returns The string representation of the complex number.
   * @see ComplexValue::toString
   */
  [[nodiscard]] std::string toString(bool formatted = true,
                                     int precision = -1) const;

  /**
   * @brief Write the complex number to a binary stream.
   * @param os The output stream to write to.
   * @see RealNumber::writeBinary
   */
  void writeBinary(std::ostream& os) const;

  /**
   * @brief Convert the Complex number to an std::complex<fp>.
   * @returns The std::complex<fp> representation of the Complex number.
   */
  [[nodiscard]] explicit operator std::complex<fp>() const noexcept;

  /**
   * @brief Convert the Complex number to a ComplexValue.
   * @returns The ComplexValue representation of the Complex number.
   */
  [[nodiscard]] explicit operator ComplexValue() const noexcept;
};

/**
 * @brief Print a complex number to a stream.
 * @param os The output stream to write to.
 * @param c The complex number to print.
 * @returns The output stream.
 */
std::ostream& operator<<(std::ostream& os, const Complex& c);

ComplexValue operator*(const Complex& c1, const ComplexValue& c2);
ComplexValue operator*(const ComplexValue& c1, const Complex& c2);
ComplexValue operator*(const Complex& c1, const Complex& c2);
ComplexValue operator*(const Complex& c1, fp real);
ComplexValue operator*(fp real, const Complex& c1);

ComplexValue operator/(const Complex& c1, const ComplexValue& c2);
ComplexValue operator/(const ComplexValue& c1, const Complex& c2);
ComplexValue operator/(const Complex& c1, const Complex& c2);
ComplexValue operator/(const Complex& c1, fp real);

} // namespace dd

/// Hash function for complex numbers.
template <> struct std::hash<dd::Complex> {
  /**
   * @brief Compute the hash value for a complex number.
   * @details Reinterprets the pointers to the real and imaginary part as
   * integers and computes the hash value for those. Afterwards, the two hash
   * values are combined.
   * @param c The complex number to compute the hash value for.
   * @returns The hash value.
   * @see dd::murmur64
   * @see dd::combineHash
   */
  std::size_t operator()(dd::Complex const& c) const noexcept {
    const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(c.r));
    const auto h2 = dd::murmur64(reinterpret_cast<std::size_t>(c.i));
    return qc::combineHash(h1, h2);
  }
};
