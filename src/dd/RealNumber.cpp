/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/RealNumber.hpp"

#include "dd/DDDefinitions.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <ostream>

namespace dd {

static constexpr std::uintptr_t LSB = 1U;
static constexpr std::uintptr_t LSB2 = 2U;

RealNumber* RealNumber::next() const noexcept {
  const std::uintptr_t ptr = reinterpret_cast<std::uintptr_t>(next_) & ~LSB2;
  return ptr == 0U ? nullptr : reinterpret_cast<RealNumber*>(ptr);
}

RealNumber* RealNumber::getAlignedPointer(const RealNumber* e) noexcept {
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) &
                                       ~LSB);
}

RealNumber* RealNumber::getNegativePointer(const RealNumber* e) noexcept {
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) |
                                       LSB);
}

RealNumber* RealNumber::flipPointerSign(const RealNumber* e) noexcept {
  if (exactlyZero(e)) {
    return &constants::zero;
  }
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) ^
                                       LSB);
}

bool RealNumber::isNegativePointer(const RealNumber* e) noexcept {
  return (reinterpret_cast<std::uintptr_t>(e) & LSB) != 0U;
}

void RealNumber::mark() noexcept {
  next_ = reinterpret_cast<RealNumber*>(
      reinterpret_cast<std::uintptr_t>(next_) | LSB2);
}

void RealNumber::unmark() noexcept {
  next_ = reinterpret_cast<RealNumber*>(
      reinterpret_cast<std::uintptr_t>(next_) & ~LSB2);
}

bool RealNumber::marked(const RealNumber* e) noexcept {
  return (reinterpret_cast<std::uintptr_t>(e->next_) & LSB2) == LSB2;
}

fp RealNumber::val(const RealNumber* e) noexcept {
  assert(e != nullptr);
  if (isNegativePointer(e)) {
    return -getAlignedPointer(e)->value;
  }
  return e->value;
}

bool RealNumber::approximatelyEquals(const fp left, const fp right) noexcept {
  return std::abs(left - right) <= eps;
}

bool RealNumber::approximatelyEquals(const RealNumber* left,
                                     const RealNumber* right) noexcept {
  return left == right || approximatelyEquals(val(left), val(right));
}

bool RealNumber::approximatelyZero(const fp e) noexcept {
  return std::abs(e) <= eps;
}

bool RealNumber::approximatelyZero(const RealNumber* e) noexcept {
  return e == &constants::zero || approximatelyZero(val(e));
}

void RealNumber::writeBinary(const RealNumber* e, std::ostream& os) {
  const auto temp = val(e);
  writeBinary(temp, os);
}

void RealNumber::writeBinary(const fp num, std::ostream& os) {
  os.write(reinterpret_cast<const char*>(&num), sizeof(fp));
}

void RealNumber::readBinary(dd::fp& num, std::istream& is) {
  is.read(reinterpret_cast<char*>(&num), sizeof(fp));
}

namespace constants {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
RealNumber zero{{nullptr}, 0.};
RealNumber one{{nullptr}, 1.};
RealNumber sqrt2over2{{nullptr}, SQRT2_2};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace constants
} // namespace dd
