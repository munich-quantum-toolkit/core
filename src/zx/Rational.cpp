/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "zx/Rational.hpp"

#include "zx/ZXDefinitions.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace zx {

PiRational::PiRational(double val) {
  if (std::abs(val) < PARAMETER_TOLERANCE) {
    return;
  }

  const double multPi = PI / val;
  const double nearest = std::round(multPi);
  if (std::abs(nearest - multPi) < PARAMETER_TOLERANCE) {
    auto denom = static_cast<int>(nearest);
    frac = Rational(1, denom);
    modPi();
    return;
  }

  val /= PI;
  val -= 2 * static_cast<int>(val / 2);
  if (val > 1) {
    val -= 2;
  } else if (val <= -1) {
    val += 2;
  }

  frac = Rational(val * MAX_DENOM, MAX_DENOM);
  modPi();
}

PiRational& PiRational::operator+=(const PiRational& rhs) {
  frac += rhs.frac;
  modPi();
  return *this;
}
PiRational& PiRational::operator+=(const int64_t rhs) {
  frac += rhs;
  modPi();
  return *this;
}

PiRational& PiRational::operator-=(const PiRational& rhs) {
  frac -= rhs.frac;
  modPi();
  return *this;
}

PiRational& PiRational::operator-=(const int64_t rhs) {
  frac -= rhs;
  modPi();
  return *this;
}

PiRational& PiRational::operator*=(const PiRational& rhs) {
  frac *= rhs.frac;
  modPi();
  return *this;
}

PiRational& PiRational::operator*=(const int64_t rhs) {
  frac *= rhs;
  modPi();
  return *this;
}

PiRational& PiRational::operator/=(const PiRational& rhs) {
  frac /= rhs.frac;
  modPi();
  return *this;
}

PiRational& PiRational::operator/=(const int64_t rhs) {
  frac /= rhs;
  modPi();
  return *this;
}

void PiRational::modPi() {
  if (*this > 1) {
    frac = Rational(getNum() - (2 * getDenom()), getDenom());
  } else if (*this <= -1) {
    frac = Rational(getNum() + (2 * getDenom()), getDenom());
  }
  if (getNum() == 0) {
    setDenom(1);
    return;
  }
}

double PiRational::toDouble() const { return frac.convert_to<double>() * PI; }
} // namespace zx
