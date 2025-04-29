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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <limits>
#include <ostream>

namespace dd {

static constexpr std::size_t LSB = static_cast<std::uintptr_t>(1U);

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

fp RealNumber::val(const RealNumber* e) noexcept {
  assert(e != nullptr);
  if (isNegativePointer(e)) {
    return -getAlignedPointer(e)->value;
  }
  return e->value;
}

RefCount RealNumber::refCount(const RealNumber* num) noexcept {
  assert(num != nullptr);
  if (isNegativePointer(num)) {
    return getAlignedPointer(num)->ref;
  }
  return num->ref;
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

bool RealNumber::noRefCountingNeeded(const RealNumber* const num) noexcept {
  assert(!isNegativePointer(num));
  return num == nullptr || constants::isStaticNumber(num) ||
         num->ref == std::numeric_limits<RefCount>::max();
}

bool RealNumber::incRef(const dd::RealNumber* num) noexcept {
  auto* const ptr = getAlignedPointer(num);
  if (noRefCountingNeeded(ptr)) {
    return false;
  }
  ++ptr->ref;
  return true;
}

bool RealNumber::decRef(const dd::RealNumber* num) noexcept {
  auto* const ptr = getAlignedPointer(num);
  if (noRefCountingNeeded(ptr)) {
    return false;
  }
  assert(ptr->ref != 0 &&
         "Reference count of RealNumber must not be zero before decrement");
  --ptr->ref;
  return true;
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
RealNumber zero{{nullptr}, 0., 1U};
RealNumber one{{nullptr}, 1., 1U};
RealNumber sqrt2over2{{nullptr}, SQRT2_2, 1U};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace constants
} // namespace dd
