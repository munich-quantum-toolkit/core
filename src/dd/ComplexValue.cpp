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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <system_error>
#include <utility>

using llvm::createStringError;
using llvm::Error;
using llvm::StringRef;

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

void ComplexValue::writeBinary(llvm::raw_ostream& os) const {
  RealNumber::writeBinary(r, os);
  RealNumber::writeBinary(i, os);
}
Error ComplexValue::fromString(StringRef realStr, StringRef imagStr) {
  const auto parse = [](StringRef text) -> llvm::Expected<fp> {
    if (text.empty()) {
      return 0.;
    }
    llvm::APFloat value(0.);
    auto status =
        value.convertFromString(text, llvm::APFloat::rmNearestTiesToEven);
    if (!status) {
      return createStringError(
          std::make_error_code(std::errc::invalid_argument),
          llvm::toString(status.takeError()));
    }
    if ((*status & llvm::APFloat::opOverflow) != 0 ||
        ((*status & llvm::APFloat::opUnderflow) != 0 && value.isZero())) {
      return createStringError(
          std::errc::result_out_of_range,
          "Serialized number is outside the finite floating-point range.");
    }
    if (!value.isFinite()) {
      return createStringError(std::errc::invalid_argument,
                               "Serialized number must be finite.");
    }
    return value.convertToDouble();
  };
  auto real = parse(realStr);
  if (!real) {
    return real.takeError();
  }

  imagStr = imagStr.trim(" ");
  const auto hasImaginaryPart = !imagStr.empty();
  if (hasImaginaryPart) {
    imagStr = imagStr.drop_back();
  }
  const bool negative = imagStr.consume_front("-");
  if (!negative) {
    imagStr.consume_front("+");
  }
  imagStr = imagStr.ltrim(" ");
  if (hasImaginaryPart && imagStr.empty()) {
    imagStr = "1";
  }
  auto imaginary = parse(imagStr);
  if (!imaginary) {
    return imaginary.takeError();
  }
  r = *real;
  i = negative ? -*imaginary : *imaginary;
  return Error::success();
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

void ComplexValue::printFormatted(llvm::raw_ostream& os, fp num, bool imaginary,
                                  int precision) {
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
    os << toString(num, 0., false, precision) << "i";
  } else {
    os << toString(num, 0., false, precision);
  }
}
std::string ComplexValue::toString(const fp& real, const fp& imag,
                                   bool formatted, int precision) {
  std::string result;
  llvm::raw_string_ostream ss(result);
  const auto number = [precision](fp value) {
    llvm::SmallString<32> text;
    llvm::APFloat(value).toString(
        text,
        precision < 0 ? 6U : static_cast<unsigned>(std::max(1, precision)));
    return text;
  };
  const auto zero = [formatted](fp value) {
    return formatted ? RealNumber::approximatelyZero(value) : value == 0.;
  };

  if (zero(real) && zero(imag)) {
    return "0";
  }

  if (!zero(real)) {
    if (formatted) {
      printFormatted(ss, real, false, precision);
    } else {
      ss << number(real);
    }
  }
  if (!zero(imag)) {
    if (formatted) {
      if (RealNumber::approximatelyEquals(real, imag)) {
        ss << "(1+i)";
        return result;
      }
      if (RealNumber::approximatelyEquals(real, -imag)) {
        ss << "(1-i)";
        return result;
      }
      printFormatted(ss, imag, true, precision);
    } else {
      if (zero(real)) {
        ss << number(imag);
      } else {
        if (imag > 0.) {
          ss << "+";
        }
        ss << number(imag);
      }
      ss << "i";
    }
  }

  return result;
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

// Computes an approximation of ac+bd
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
  // Trade cancellation accuracy for throughput in this hot path.
  const auto r = std::fma(-c1.i, c2.i, c1.r * c2.r);
  const auto i = std::fma(c1.i, c2.r, c1.r * c2.i);
  return {r, i};
}

ComplexValue operator/(const ComplexValue& c1, fp r) {
  return {c1.r / r, c1.i / r};
}

ComplexValue operator/(const ComplexValue& c1, const ComplexValue& c2) {
  // Avoid squaring a scalar denominator outside the squared floating range.
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

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ComplexValue& c) {
  return os << ComplexValue::toString(c.r, c.i);
}

} // namespace dd

std::size_t std::hash<dd::ComplexValue>::operator()(
    const dd::ComplexValue& c) const noexcept {
  const auto hashComponent = [](dd::fp value) {
    const auto rounded = std::round(value / dd::RealNumber::eps);
    return rounded == 0. ? size_t{0} : std::hash<dd::fp>{}(rounded);
  };
  return dd::combineHash(hashComponent(c.r), hashComponent(c.i));
}
