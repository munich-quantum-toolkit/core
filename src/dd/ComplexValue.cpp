/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComplexValue.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/RealNumber.hpp"

#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <ios>
#include <istream>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace dd {
bool ComplexValue::operator==(const ComplexValue& other) const noexcept {
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  return r == other.r && i == other.i;
}

bool ComplexValue::operator!=(const ComplexValue& other) const noexcept {
  return !operator==(other);
}

bool ComplexValue::approximatelyEquals(const ComplexValue& c) const noexcept {
  return RealNumber::approximatelyEquals(r, c.r) &&
         RealNumber::approximatelyEquals(i, c.i);
}

bool ComplexValue::approximatelyZero() const noexcept {
  return RealNumber::approximatelyZero(r) && RealNumber::approximatelyZero(i);
}

void ComplexValue::writeBinary(std::ostream& os) const {
  RealNumber::writeBinary(r, os);
  RealNumber::writeBinary(i, os);
}

void ComplexValue::readBinary(std::istream& is) {
  RealNumber::readBinary(r, is);
  RealNumber::readBinary(i, is);
}

mlir::FailureOr<ComplexValue> ComplexValue::parse(std::string_view text) {
  ComplexValue value;
  auto const readPart = [](std::string_view& input, fp& part) {
    std::istringstream stream{std::string(input)};
    stream.imbue(std::locale::classic());
    stream >> std::noskipws >> part;
    /// libc++ reports underflow even when the subnormal result is
    /// representable.
    if ((stream.fail() && std::fpclassify(part) != FP_SUBNORMAL) ||
        !std::isfinite(part)) {
      return false;
    }
    const auto consumed = static_cast<size_t>(
        stream.rdbuf()->pubseekoff(0, std::ios_base::cur, std::ios_base::in));
    const auto number = input.substr(0, consumed);
    if (number.find_first_not_of("0123456789eE+-.") != std::string_view::npos ||
        (std::fpclassify(part) == FP_ZERO &&
         number.substr(0, number.find_first_of("eE"))
                 .find_first_of("123456789") != std::string_view::npos)) {
      return false;
    }
    input.remove_prefix(consumed);
    return true;
  };
  auto const imaginaryUnit = [](std::string_view input) {
    return input == "i" || input == "I";
  };
  if (text.empty()) {
    return value;
  }
  /// A real prefix can instead be the coefficient of a purely imaginary value.
  auto remaining = text;
  fp first = 0.;
  if (readPart(remaining, first)) {
    if (remaining.empty()) {
      return ComplexValue{first};
    }
    if (imaginaryUnit(remaining)) {
      return ComplexValue{0., first};
    }
    value.r = first;
    text = remaining;
  }
  /// The serialized format permits spaces around the imaginary sign.
  std::string imaginary(text);
  std::erase(imaginary, ' ');
  text = imaginary;
  if (text.starts_with('+')) {
    text.remove_prefix(1);
  }
  if (imaginaryUnit(text)) {
    value.i = 1.;
    return value;
  }
  if (text.starts_with('-') && imaginaryUnit(text.substr(1))) {
    value.i = -1.;
    return value;
  }
  if (!readPart(text, value.i) || !imaginaryUnit(text)) {
    return ::mqt::emitError("Invalid serialized complex number.",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return value;
}

std::pair<std::uint64_t, std::uint64_t>
ComplexValue::getLowestFraction(const fp x,
                                const std::uint64_t maxDenominator) {
  assert(x >= 0.);

  std::pair<std::uint64_t, std::uint64_t> lowerBound{0U, 1U};
  std::pair<std::uint64_t, std::uint64_t> upperBound{1U, 0U};

  while ((lowerBound.second <= maxDenominator) &&
         (upperBound.second <= maxDenominator)) {
    auto const num = lowerBound.first + upperBound.first;
    auto const den = lowerBound.second + upperBound.second;
    auto const median = static_cast<fp>(num) / static_cast<fp>(den);
    if (std::abs(x - median) <= RealNumber::eps) {
      if (den <= maxDenominator) {
        return std::pair{num, den};
      }
      if (upperBound.second > lowerBound.second) {
        return upperBound;
      }
      return lowerBound;
    }
    if (x > median) {
      lowerBound = {num, den};
    } else {
      upperBound = {num, den};
    }
  }
  if (lowerBound.second > maxDenominator) {
    return upperBound;
  }
  return lowerBound;
}

void ComplexValue::printFormatted(std::ostream& os, fp num, bool imaginary) {
  if (std::signbit(num)) {
    os << "-";
    num = -num;
  } else if (imaginary) {
    os << "+";
  }

  if (RealNumber::approximatelyZero(num)) {
    os << "0" << (imaginary ? "i" : "");
    return;
  }

  const auto absnum = std::abs(num);
  auto fraction = getLowestFraction(absnum);
  auto approx =
      static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);

  // suitable fraction a/b found
  if (const auto error = absnum - approx;
      RealNumber::approximatelyZero(error)) {
    if (fraction.first == 1U && fraction.second == 1U) {
      os << (imaginary ? "i" : "1");
    } else if (fraction.second == 1U) {
      os << fraction.first << (imaginary ? "i" : "");
    } else if (fraction.first == 1U) {
      os << (imaginary ? "i" : "1") << "/" << fraction.second;
    } else {
      os << fraction.first << (imaginary ? "i" : "") << "/" << fraction.second;
    }

    return;
  }

  const auto abssqrt = absnum / SQRT2_2;
  fraction = getLowestFraction(abssqrt);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  // suitable fraction a/(b * sqrt(2)) found
  if (const auto error = abssqrt - approx;
      RealNumber::approximatelyZero(error)) {

    if (fraction.first == 1U && fraction.second == 1U) {
      os << (imaginary ? "i" : "1") << "/√2";
    } else if (fraction.second == 1U) {
      os << fraction.first << (imaginary ? "i" : "") << "/√2";
    } else if (fraction.first == 1U) {
      os << (imaginary ? "i" : "1") << "/(" << fraction.second << "√2)";
    } else {
      os << fraction.first << (imaginary ? "i" : "") << "/(" << fraction.second
         << "√2)";
    }
    return;
  }

  const auto abspi = absnum / PI;
  fraction = getLowestFraction(abspi);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  // suitable fraction a/b π found
  if (const auto error = abspi - approx; RealNumber::approximatelyZero(error)) {
    const std::string imagUnit = imaginary ? "i" : "";

    if (fraction.first == 1U && fraction.second == 1U) {
      os << "π" << imagUnit;
    } else if (fraction.second == 1U) {
      os << fraction.first << "π" << imagUnit;
    } else if (fraction.first == 1U) {
      os << "π" << imagUnit << "/" << fraction.second;
    } else {
      os << fraction.first << "π" << imagUnit << "/" << fraction.second;
    }
    return;
  }

  if (imaginary) { // default
    os << num << "i";
  } else {
    os << num;
  }
}

std::string ComplexValue::toString(const fp& real, const fp& imag,
                                   bool formatted, int precision) {
  std::ostringstream ss{};
  const auto zero = [formatted](fp value) {
    return formatted ? RealNumber::approximatelyZero(value) : value == 0.;
  };

  if (precision >= 0) {
    ss << std::setprecision(precision);
  }
  if (zero(real) && zero(imag)) {
    return "0";
  }

  if (!zero(real)) {
    if (formatted) {
      printFormatted(ss, real);
    } else {
      ss << real;
    }
  }
  if (!zero(imag)) {
    if (formatted) {
      if (RealNumber::approximatelyEquals(real, imag)) {
        ss << "(1+i)";
        return ss.str();
      }
      if (RealNumber::approximatelyEquals(real, -imag)) {
        ss << "(1-i)";
        return ss.str();
      }
      printFormatted(ss, imag, true);
    } else {
      if (zero(real)) {
        ss << imag;
      } else {
        if (imag > 0.) {
          ss << "+";
        }
        ss << imag;
      }
      ss << "i";
    }
  }

  return ss.str();
}

ComplexValue& ComplexValue::operator+=(const ComplexValue& rhs) noexcept {
  r += rhs.r;
  i += rhs.i;
  return *this;
}

ComplexValue& ComplexValue::operator*=(const fp& real) noexcept {
  r *= real;
  i *= real;
  return *this;
}

ComplexValue operator+(const ComplexValue& c1, const ComplexValue& c2) {
  return {c1.r + c2.r, c1.i + c2.i};
}

ComplexValue operator*(const ComplexValue& c1, fp r) {
  return {c1.r * r, c1.i * r};
}

ComplexValue operator*(fp r, const ComplexValue& c1) {
  return {c1.r * r, c1.i * r};
}

/// Computes an approximation of ac+bd
namespace {
fp kahan(const fp a, const fp b, const fp c, const fp d) {
  // w = RN(b * d)
  const auto w = b * d;
  // e = RN(b * d - w)
  const auto e = std::fma(b, d, -w);
  // f = RN(a * c + w)
  const auto f = std::fma(a, c, w);
  // g = RN(f + e)
  return f + e;
}
} // namespace

ComplexValue operator*(const ComplexValue& c1, const ComplexValue& c2) {
  // Implements the CMulKahan algorithm from https://hal.science/hal-01512760v2
  // p1 = RN(c1.r * c2.r)
  // R = RN(RN(p1 - c1.i * c2.i) + RN(c1.r * c2.r - p1))
  const auto r = kahan(-c1.i, c1.r, c2.i, c2.r);
  // p3 = RN(c1.r * c2.i)
  // I = RN(RN(p3 + c1.i * c2.r) + RN(c1.r * c2.i - p3))
  const auto i = kahan(c1.i, c1.r, c2.r, c2.i);
  return {r, i};
}

ComplexValue operator/(const ComplexValue& c1, fp r) {
  return {c1.r / r, c1.i / r};
}

ComplexValue operator/(const ComplexValue& c1, const ComplexValue& c2) {
  /// Avoid squaring a scalar denominator outside the squared floating range.
  if (c2.i == 0.) {
    return c1 / c2.r;
  }
  if (c2.r == 0.) {
    return {c1.i / c2.i, -c1.r / c2.i};
  }
  // Implements the CompDivT algorithm from
  // https://ens-lyon.hal.science/ensl-00734339v2

  // Selects the denominator with the smallest relative error bound
  const auto d = std::abs(c2.i) <= std::abs(c2.r)
                     ? std::fma(c2.r, c2.r, c2.i * c2.i)
                     : std::fma(c2.i, c2.i, c2.r * c2.r);
  // evaluates c1.r * c2.r + c1.i * c2.i
  const auto gr = kahan(c1.r, c1.i, c2.r, c2.i);
  // evaluates c1.i * c2.r - c1.r * c2.i
  const auto gi = kahan(c1.i, -c1.r, c2.r, c2.i);
  return {gr / d, gi / d};
}

std::ostream& operator<<(std::ostream& os, const ComplexValue& c) {
  return os << ComplexValue::toString(c.r, c.i);
}
} // namespace dd

std::size_t std::hash<dd::ComplexValue>::operator()(
    const dd::ComplexValue& c) const noexcept {
  const auto h1 = std::hash<dd::fp>{}(std::round(c.r / dd::RealNumber::eps));
  const auto h2 = std::hash<dd::fp>{}(std::round(c.i / dd::RealNumber::eps));
  return dd::combineHash(h1, h2);
}
