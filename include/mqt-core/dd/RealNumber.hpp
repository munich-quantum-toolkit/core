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
#include "dd/LinkedListBase.hpp"
#include "dd/mqt_core_dd_export.h"

#include <istream>
#include <limits>
#include <ostream>

namespace dd {
/**
 * @brief A struct for representing real numbers as part of the DD package.
 * @details Consists of a floating point number (the value) and a next pointer
 * (used for chaining entries). Numbers are marked for garbage collection via
 * the second least significant bit of pointers referencing them.
 * @note Due to the way the sign of the value is encoded, special care has to
 * be taken when accessing the value. The static functions in this struct
 * provide safe access to the value of a RealNumber* pointer.
 */
struct RealNumber final : LLBase {
  static constexpr std::uintptr_t MARK_BIT = 1U << 1U;

  /// Getter for the next object.
  [[nodiscard]] RealNumber* next() const noexcept {
    return reinterpret_cast<RealNumber*>(
        reinterpret_cast<std::uintptr_t>(next_) & ~MARK_BIT);
  }

  void markEntry() noexcept {
    next_ = reinterpret_cast<RealNumber*>(
        reinterpret_cast<std::uintptr_t>(next_) | MARK_BIT);
  }

  void unmarkEntry() noexcept {
    next_ = reinterpret_cast<RealNumber*>(
        reinterpret_cast<std::uintptr_t>(next_) & ~MARK_BIT);
  }

  [[nodiscard]] bool isMarked() const noexcept {
    return (reinterpret_cast<std::uintptr_t>(next_) & MARK_BIT) != 0U;
  }

  /**
   * @brief Check whether the number points to the zero number.
   * @param e The number to check.
   * @returns Whether the number points to zero.
   */
  [[nodiscard]] static constexpr bool exactlyZero(const RealNumber* e) noexcept;

  /**
   * @brief Check whether the number points to the one number.
   * @param e The number to check.
   * @returns Whether the number points to one.
   */
  [[nodiscard]] static constexpr bool exactlyOne(const RealNumber* e) noexcept;

  /**
   * @brief Check whether the number points to the sqrt(2)/2 = 1/sqrt(2) number.
   * @param e The number to check.
   * @returns Whether the number points to negative one.
   */
  [[nodiscard]] static constexpr bool
  exactlySqrt2over2(const RealNumber* e) noexcept;

  /**
   * @brief Get the value of the number.
   * @param e The number to get the value for.
   * @returns The value of the number.
   * @note This function accounts for the sign of the number embedded in the
   * memory address of the number.
   */
  [[nodiscard]] static fp val(const RealNumber* e) noexcept;

  /**
   * @brief Check whether two floating point numbers are approximately equal.
   * @details This function checks whether two floating point numbers are
   * approximately equal. The two numbers are considered approximately equal
   * if the absolute difference between them is smaller than a small value
   * (TOLERANCE). This function is used to compare floating point numbers
   * stored in the table.
   * @param left The first floating point number.
   * @param right The second floating point number.
   * @returns Whether the two floating point numbers are approximately equal.
   */
  [[nodiscard]] static bool approximatelyEquals(fp left, fp right) noexcept;

  /**
   * @brief Check whether two numbers are approximately equal.
   * @details This function checks whether two numbers are approximately
   * equal. Two numbers are considered approximately equal if they point to
   * the same number or if the values of the numbers are approximately equal.
   * @param left The first number.
   * @param right The second number.
   * @returns Whether the two numbers are approximately equal.
   * @see approximatelyEquals(fp, fp)
   */
  [[nodiscard]] static bool
  approximatelyEquals(const RealNumber* left, const RealNumber* right) noexcept;

  /**
   * @brief Check whether a floating point number is approximately zero.
   * @param e The floating point number to check.
   * @returns Whether the floating point number is approximately zero.
   */
  [[nodiscard]] static bool approximatelyZero(fp e) noexcept;

  /**
   * @brief Check whether a number is approximately zero.
   * @param e The number to check.
   * @returns Whether the number is approximately zero.
   * @see approximatelyZero(fp)
   */
  [[nodiscard]] static bool approximatelyZero(const RealNumber* e) noexcept;

  /**
   * @brief Write a binary representation of the number to a stream.
   * @param e The number to write.
   * @param os The stream to write to.
   */
  static void writeBinary(const RealNumber* e, std::ostream& os);

  /**
   * @brief Write a binary representation of a floating point number to a
   * @param num The number to write.
   * @param os The stream to write to.
   */
  static void writeBinary(fp num, std::ostream& os);

  /**
   * @brief Read a binary representation of a number from a stream.
   * @param num The number to read into.
   * @param is The stream to read from.
   */
  static void readBinary(fp& num, std::istream& is);

  /**
   * @brief Get an aligned pointer to the number.
   * @details Since the least significant bit of the memory address of the
   * number is used to encode the sign of the value, the pointer to the number
   * might not be aligned. This function returns an aligned pointer to the
   * number.
   * @param e The number to get the aligned pointer for.
   * @returns An aligned pointer to the number.
   */
  [[nodiscard]] static RealNumber*
  getAlignedPointer(const RealNumber* e) noexcept;

  /**
   * @brief Get a pointer to the number with a negative sign.
   * @details Since the least significant bit of the memory address of the
   * number is used to encode the sign of the value, this function just sets
   * the least significant bit of the memory address of the number to 1.
   * @param e The number to get the negative pointer for.
   * @returns A negative pointer to the number.
   */
  [[nodiscard]] static RealNumber*
  getNegativePointer(const RealNumber* e) noexcept;

  /**
   * @brief Check whether the number is a negative pointer.
   * @param e The number to check.
   * @returns Whether the number is a negative pointer.
   */
  [[nodiscard]] static bool isNegativePointer(const RealNumber* e) noexcept;

  /**
   * @brief Flip the sign of the number pointer.
   * @param e The number to flip the sign of.
   * @returns The number with the sign flipped.
   * @note This function does not change the sign of the value of the number.
   * It rather changes the sign of the pointer to the number.
   * @note We do not consider negative zero here, since it is not used in the
   * DD package. There only exists one zero number, which is positive.
   */
  [[nodiscard]] static RealNumber*
  flipPointerSign(const RealNumber* e) noexcept;

  /**
   * @brief Mark a real number pointer for garbage collection.
   * @param p Pointer to mark.
   * @return The marked pointer.
   */
  [[nodiscard]] static RealNumber*
  getMarkedPointer(const RealNumber* p) noexcept;

  /**
   * @brief Check whether a real number pointer is marked.
   * @param p Pointer to check.
   * @return True if the pointer is marked.
   */
  [[nodiscard]] static bool isMarkedPointer(const RealNumber* p) noexcept;

  /**
   * @brief Clear the mark bit from a pointer.
   * @param p Pointer to clear.
   * @return The unmarked pointer.
   */
  [[nodiscard]] static RealNumber* clearMark(const RealNumber* p) noexcept;

  /**
   * @brief The value of the number.
   * @details The value of the number is a floating point number. The sign of
   * the value is encoded in the least significant bit of the memory address
   * of the number. As a consequence, values stored here are always
   * non-negative. The sign of the value as well as the value itself can be
   * accessed using the static functions of this struct.
   */
  fp value{};

  /// numerical tolerance to be used for floating point values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static inline fp eps = std::numeric_limits<dd::fp>::epsilon() * 1024;
};

namespace constants {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
/// The constant zero.
MQT_CORE_DD_EXPORT extern RealNumber zero;
/// The constant one.
MQT_CORE_DD_EXPORT extern RealNumber one;
/// The constant sqrt(2)/2 = 1/sqrt(2).
MQT_CORE_DD_EXPORT extern RealNumber sqrt2over2;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/**
 * @brief Check whether a number is one of the static numbers.
 * @param e The number to check.
 * @return Whether the number is one of the static numbers.
 */
[[nodiscard]] constexpr bool isStaticNumber(const RealNumber* e) noexcept {
  return RealNumber::exactlyZero(e) || RealNumber::exactlyOne(e) ||
         RealNumber::exactlySqrt2over2(e);
}
} // namespace constants

constexpr bool RealNumber::exactlyZero(const RealNumber* e) noexcept {
  return clearMark(e) == &constants::zero;
}

constexpr bool RealNumber::exactlyOne(const RealNumber* e) noexcept {
  return clearMark(e) == &constants::one;
}

constexpr bool RealNumber::exactlySqrt2over2(const RealNumber* e) noexcept {
  return clearMark(e) == &constants::sqrt2over2;
}
} // namespace dd
