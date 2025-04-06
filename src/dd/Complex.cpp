/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Complex.hpp"

#include "dd/ComplexValue.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/RealNumber.hpp"

#include <complex>
#include <ostream>
#include <string>

namespace dd {

bool Complex::approximatelyEquals(const Complex& c) const noexcept {
  return RealNumber::approximatelyEquals(r, c.r) &&
         RealNumber::approximatelyEquals(i, c.i);
}

bool Complex::approximatelyZero() const noexcept {
  return RealNumber::approximatelyZero(r) && RealNumber::approximatelyZero(i);
}

std::string Complex::toString(bool formatted, int precision) const {
  return ComplexValue::toString(RealNumber::val(r), RealNumber::val(i),
                                formatted, precision);
}

void Complex::writeBinary(std::ostream& os) const {
  RealNumber::writeBinary(r, os);
  RealNumber::writeBinary(i, os);
}

Complex::operator std::complex<fp>() const noexcept {
  return {RealNumber::val(r), RealNumber::val(i)};
}

Complex::operator ComplexValue() const noexcept {
  return ComplexValue{RealNumber::val(r), RealNumber::val(i)};
}

std::ostream& operator<<(std::ostream& os, const Complex& c) {
  return os << c.toString();
}

ComplexValue operator*(const Complex& c1, const ComplexValue& c2) {
  return static_cast<ComplexValue>(c1) * c2;
}
ComplexValue operator*(const ComplexValue& c1, const Complex& c2) {
  return c1 * static_cast<ComplexValue>(c2);
}
ComplexValue operator*(const Complex& c1, const Complex& c2) {
  return static_cast<ComplexValue>(c1) * static_cast<ComplexValue>(c2);
}
ComplexValue operator*(const Complex& c1, const fp real) {
  return static_cast<ComplexValue>(c1) * real;
}
ComplexValue operator*(const fp real, const Complex& c1) { return c1 * real; }

ComplexValue operator/(const Complex& c1, const ComplexValue& c2) {
  return static_cast<ComplexValue>(c1) / c2;
}
ComplexValue operator/(const ComplexValue& c1, const Complex& c2) {
  return c1 / static_cast<ComplexValue>(c2);
}
ComplexValue operator/(const Complex& c1, const Complex& c2) {
  return static_cast<ComplexValue>(c1) / static_cast<ComplexValue>(c2);
}
ComplexValue operator/(const Complex& c1, const fp real) {
  return static_cast<ComplexValue>(c1) / real;
}
} // namespace dd
