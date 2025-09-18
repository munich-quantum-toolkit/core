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

#if defined(GMP)
#include <boost/multiprecision/gmp.hpp> // IWYU pragma: keep
using Rational = boost::multiprecision::mpq_rational;
using BigInt = boost::multiprecision::mpz_int;
#else
#include <boost/multiprecision/cpp_int.hpp> // IWYU pragma: keep
#include <boost/multiprecision/fwd.hpp>
using Rational = boost::multiprecision::cpp_rational;
using BigInt = boost::multiprecision::cpp_int;
#endif

#include <boost/multiprecision/rational_adaptor.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace zx {

/*
 * @brief Representation of fractions as multiples of pi
 * @details Rationals can only have values in the half-open interval (-1,1],
 * corresponding to the interval (-pi, pi]
 */
class PiRational {
public:
  /**
   * @brief Default constructor initializes the rational to 0/1.
   */
  PiRational() = default;

  /**
   * @brief Construct a PiRational from numerator and denominator.
   * @details The input fraction is already assumed to be in multiples of Pi.
   * For example, the fraction 1/2 corresponds to Pi/2.
   * @param num Numerator of the fraction.
   * @param denom Denominator of the fraction.
   */
  explicit PiRational(int64_t num, int64_t denom) : frac(num, denom) {
    modPi();
  }
  explicit PiRational(const BigInt& num, const BigInt& denom)
      : frac(num, denom) {
    modPi();
  }

  /**
   * @brief Construct a PiRational from numerator. Denominator is assumed to
   * be 1.
   * @details The input numerator is already assumed to be in multiples of Pi.
   * For example a numerator of 1 corresponds to a fraction Pi/1.
   * @param num Numerator of the fraction.
   */
  explicit PiRational(int64_t num) : frac(num, 1) { modPi(); }

  /**
   * @brief Construct a PiRational from a double.
   * @details The input double is approximated as a fraction of Pi within a
   * tolerance of 1e-13.
   * @param val Double value to be approximated.
   */
  explicit PiRational(double val);

  PiRational& operator+=(const PiRational& rhs);
  PiRational& operator+=(int64_t rhs);

  PiRational& operator-=(const PiRational& rhs);
  PiRational& operator-=(int64_t rhs);

  PiRational& operator*=(const PiRational& rhs);
  PiRational& operator*=(int64_t rhs);

  PiRational& operator/=(const PiRational& rhs);
  PiRational& operator/=(int64_t rhs);

  /**
   * @brief Check if the fraction is an integer, i.e., the denominator is 1.
   * @return True if the fraction is an integer, false otherwise.
   */
  [[nodiscard]] bool isInteger() const {
    return boost::multiprecision::denominator(frac) == 1;
  }

  /**
   * @brief Check if the fraction is zero, i,e, the numerator is 0.
   * @return True if the fraction is zero, false otherwise.
   */
  [[nodiscard]] bool isZero() const {
    return boost::multiprecision::numerator(frac) == 0;
  }

  /**
   * @brief Get the denominator of the fraction.
   * @return Denominator of the fraction.
   */
  [[nodiscard]] BigInt getDenom() const {
    return boost::multiprecision::denominator(frac);
  }

  /**
   * @brief Get the numerator of the fraction.
   * @return Numerator of the fraction.
   */
  [[nodiscard]] BigInt getNum() const {
    return boost::multiprecision::numerator(frac);
  }

  /**
   * @brief Convert the fraction to a double.
   * @details The result is not taken mod Pi. Converting 1/1 will return an
   * approximation of Pi.
   * @return Double value of the fraction.
   */
  [[nodiscard]] double toDouble() const;

  /**
   * @brief Convert the fraction to a double mod Pi.
   * @details The result is taken mod Pi. Converting 1/1 will return 1.0.
   * @return Double value of the fraction.
   */
  [[nodiscard]] double toDoubleDivPi() const {
    return frac.convert_to<double>();
  }

  /**
   * @brief Check if the fraction is close to a double value within a tolerance.
   * @details The comparison is not done mod Pi. So if the fraction is 1/1
   * isClose(1.0, 1e-13) will return true.
   * @param x Double value to compare to.
   * @param tolerance Tolerance for the comparison.
   * @return True if the fraction is close to the double value, false otherwise.
   */
  [[nodiscard]] bool isClose(const double x, const double tolerance) const {
    return std::abs(toDouble() - x) < tolerance;
  }

  /**
   * @brief Check if the fraction is close to a double value within a tolerance
   * mod Pi.
   * @details The comparison is done mod Pi. So if the fraction is 1/1
   * isCloseDivPi(1.0, 1e-13) will return false, but isCloseDivPi(3.14159,
   * 1e-14) will return true.
   * @param x Double value to compare to.
   * @param tolerance Tolerance for the comparison.
   * @return True if the fraction is close to the double value divided by Pi,
   * false otherwise.
   */
  [[nodiscard]] bool isCloseDivPi(const double x,
                                  const double tolerance) const {
    return std::abs(toDoubleDivPi() - x) < tolerance;
  }

  /**
   * @brief Get the double value of the fraction.
   * @details The result is not taken mod Pi. Converting 1/1 will return 1.0.
   * @return Double value of the fraction.
   */
  explicit operator double() const { return this->toDouble(); }

private:
  Rational frac;

  void modPi();

  void setNum(const BigInt& num) {
    boost::multiprecision::numerator(frac) = num;
  }

  void setDenom(const BigInt& denom) const {
    boost::multiprecision::denominator(frac) = denom;
  }
};

inline PiRational operator-(const PiRational& rhs) {
  return PiRational(-rhs.getNum(), rhs.getDenom());
}
inline PiRational operator+(PiRational lhs, const PiRational& rhs) {
  lhs += rhs;
  return lhs;
}
inline PiRational operator+(PiRational lhs, const int64_t rhs) {
  lhs += rhs;
  return lhs;
}
inline PiRational operator+(const int64_t lhs, PiRational rhs) {
  rhs += lhs;
  return rhs;
}

inline PiRational operator-(PiRational lhs, const PiRational& rhs) {
  lhs -= rhs;
  return lhs;
}
inline PiRational operator-(PiRational lhs, const int64_t rhs) {
  lhs -= rhs;
  return lhs;
}
inline PiRational operator-(const int64_t lhs, PiRational rhs) {
  rhs -= lhs;
  return rhs;
}

inline PiRational operator*(PiRational lhs, const PiRational& rhs) {
  lhs *= rhs;
  return lhs;
}
inline PiRational operator*(PiRational lhs, const int64_t rhs) {
  lhs *= rhs;
  return lhs;
}
inline PiRational operator*(const int64_t lhs, PiRational rhs) {
  rhs *= lhs;
  return rhs;
}

inline PiRational operator/(PiRational lhs, const PiRational& rhs) {
  lhs /= rhs;
  return lhs;
}
inline PiRational operator/(PiRational lhs, const int64_t rhs) {
  lhs /= rhs;
  return lhs;
}
inline PiRational operator/(const int64_t lhs, PiRational rhs) {
  rhs /= lhs;
  return rhs;
}

inline bool operator<(const PiRational& lhs, const PiRational& rhs) {
  return (lhs.getNum() * rhs.getDenom()) < (rhs.getNum() * lhs.getDenom());
}

inline bool operator<(const PiRational& lhs, const int64_t rhs) {
  return lhs.getNum() < (rhs * lhs.getDenom());
}

inline bool operator<(const int64_t lhs, const PiRational& rhs) {
  return (lhs * rhs.getDenom()) < rhs.getNum();
}

inline bool operator<=(const PiRational& lhs, const PiRational& rhs) {
  return (lhs.getNum() * rhs.getDenom()) <= (rhs.getNum() * lhs.getDenom());
}

inline bool operator<=(const PiRational& lhs, const int64_t rhs) {
  return lhs.getNum() <= (rhs * lhs.getDenom());
}

inline bool operator<=(const int64_t lhs, const PiRational& rhs) {
  return (lhs * rhs.getDenom()) <= rhs.getNum();
}

inline bool operator>(const PiRational& lhs, const PiRational& rhs) {
  return rhs < lhs;
}

inline bool operator>(const PiRational& lhs, const int64_t rhs) {
  return rhs < lhs;
}

inline bool operator>(const int64_t lhs, const PiRational& rhs) {
  return rhs < lhs;
}

inline bool operator>=(const PiRational& lhs, const PiRational& rhs) {
  return rhs <= lhs;
}

inline bool operator>=(const PiRational& lhs, const int64_t rhs) {
  return rhs <= lhs;
}

inline bool operator>=(const int64_t lhs, const PiRational& rhs) {
  return rhs <= lhs;
}

inline bool operator==(const PiRational& lhs, const PiRational& rhs) {
  return lhs.getNum() == rhs.getNum() && lhs.getDenom() == rhs.getDenom();
}

inline bool operator==(const PiRational& lhs, const int64_t rhs) {
  return lhs.getNum() == rhs && lhs.getDenom() == 1;
}

inline bool operator==(const int64_t lhs, const PiRational& rhs) {
  return rhs == lhs;
}

inline bool operator!=(const PiRational& lhs, const PiRational& rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const PiRational& lhs, const int64_t rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const int64_t lhs, const PiRational& rhs) {
  return !(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, const zx::PiRational& rhs) {
  os << rhs.getNum() << "/" << rhs.getDenom();
  return os;
}

} // namespace zx
