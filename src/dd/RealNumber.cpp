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

static constexpr std::uintptr_t NEG_FLAG = (1UL << 0);
static constexpr std::uintptr_t MARK_FLAG = (1UL << 1);
static constexpr std::uintptr_t MARK_IMMORTAL = (1UL << 2);

RealNumber* RealNumber::next() const noexcept {
  return RealNumber::getAlignedPointer(reinterpret_cast<RealNumber*>(next_));
}

RealNumber* RealNumber::getAlignedPointer(const RealNumber* e) noexcept {
  return reinterpret_cast<RealNumber*>(
      reinterpret_cast<std::uintptr_t>(e) &
      (~(NEG_FLAG | MARK_FLAG | MARK_IMMORTAL)));
}

RealNumber* RealNumber::getNegativePointer(const RealNumber* e) noexcept {
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) |
                                       NEG_FLAG);
}

RealNumber* RealNumber::flipPointerSign(const RealNumber* e) noexcept {
  if (exactlyZero(e)) {
    return &constants::zero;
  }
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) ^
                                       NEG_FLAG);
}

bool RealNumber::isNegativePointer(const RealNumber* e) noexcept {
  return (reinterpret_cast<std::uintptr_t>(e) & NEG_FLAG) != 0U;
}

void RealNumber::mark(RealNumber* e) noexcept {
  RealNumber* p = isNegativePointer(e) ? getAlignedPointer(e) : e;
  p->next_ = reinterpret_cast<RealNumber*>(
      reinterpret_cast<std::uintptr_t>(p->next_) | MARK_FLAG);
}

void RealNumber::unmark(RealNumber* e) noexcept {
  RealNumber* p = isNegativePointer(e) ? getAlignedPointer(e) : e;
  p->next_ = reinterpret_cast<RealNumber*>(
      reinterpret_cast<std::uintptr_t>(p->next_) & ~MARK_FLAG);
}

bool RealNumber::isMarked(const RealNumber* e) noexcept {
  const RealNumber* p = isNegativePointer(e) ? getAlignedPointer(e) : e;
  return (reinterpret_cast<std::uintptr_t>(p->next_) & MARK_FLAG) != 0U;
}

void RealNumber::immortalize(RealNumber* e) noexcept {
  RealNumber* p = isNegativePointer(e) ? getAlignedPointer(e) : e;
  p->next_ = reinterpret_cast<RealNumber*>(
      reinterpret_cast<std::uintptr_t>(p->next_) | MARK_IMMORTAL);
}

bool RealNumber::isImmortal(const RealNumber* e) noexcept {
  const RealNumber* p = isNegativePointer(e) ? getAlignedPointer(e) : e;
  return (reinterpret_cast<std::uintptr_t>(p->next_) & MARK_IMMORTAL) != 0U;
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
