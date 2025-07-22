/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/ComplexValue.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/RealNumber.hpp"
#include "ir/Definitions.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
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

void ComplexValue::fromString(const std::string& realStr, std::string imagStr) {
  r = realStr.empty() ? 0. : std::stod(realStr);

  std::erase(imagStr, ' ');
  std::erase(imagStr, 'i');
  if (imagStr == "+" || imagStr == "-") {
    imagStr = imagStr + "1";
  }
  i = imagStr.empty() ? 0. : std::stod(imagStr);
}

std::pair<std::uint64_t, std::uint64_t>
ComplexValue::getLowestFraction(const fp x,
                                const std::uint64_t maxDenominator) {
  assert(x >= 0.);

  std::pair<std::uint64_t, std::uint64_t> lowerBound{0U, 1U};
  std::pair<std::uint64_t, std::uint64_t> upperBound{1U, 0U};

  while ((lowerBound.second <= maxDenominator) &&
         (upperBound.second <= maxDenominator)) {
    auto num = lowerBound.first + upperBound.first;
    auto den = lowerBound.second + upperBound.second;
    auto median = static_cast<fp>(num) / static_cast<fp>(den);
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

  if (precision >= 0) {
    ss << std::setprecision(precision);
  }
  if (RealNumber::approximatelyZero(real) &&
      RealNumber::approximatelyZero(imag)) {
    return "0";
  }

  if (!RealNumber::approximatelyZero(real)) {
    if (formatted) {
      printFormatted(ss, real);
    } else {
      ss << real;
    }
  }
  if (!RealNumber::approximatelyZero(imag)) {
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
      if (RealNumber::approximatelyZero(real)) {
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
  // performs the division
  return {gr / d, gi / d};
}

std::ostream& operator<<(std::ostream& os, const ComplexValue& c) {
  return os << ComplexValue::toString(c.r, c.i);
}
} // namespace dd

namespace std {
std::size_t
hash<dd::ComplexValue>::operator()(const dd::ComplexValue& c) const noexcept {
  const auto h1 = dd::murmur64(
      static_cast<std::size_t>(std::round(c.r / dd::RealNumber::eps)));
  const auto h2 = dd::murmur64(
      static_cast<std::size_t>(std::round(c.i / dd::RealNumber::eps)));
  return qc::combineHash(h1, h2);
}
} // namespace std
