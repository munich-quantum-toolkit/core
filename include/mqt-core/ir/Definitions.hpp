/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Common definitions used throughout the library.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qc {
/**
 * @brief Type alias for qubit indices.
 * @details This type (alias) is used to represent qubit indices in the library.
 * It has been chosen to be an unsigned 32-bit integer to allow for up to
 * 4,294,967,295 qubits, which should be enough for most use cases.
 */
using Qubit = std::uint32_t;
/**
 * @brief Type alias for classical bit indices.
 * @details This type (alias) is used to represent classical bit indices in the
 * library. The choice of 64-bits is arbitrary and can be changed if necessary.
 */
using Bit = std::uint64_t;

/// A type alias for a vector of qubits which are supposed to act as targets.
using Targets = std::vector<Qubit>;

/// Floating-point type used throughout the library
using fp = double;

/// A constant for the value of \f$\pi\f$.
static constexpr auto PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);
/// A constant for the value of \f$\frac{\pi}{2}\f$.
static constexpr auto PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
/// A constant for the value of \f$\frac{\pi}{4}\f$.
static constexpr auto PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);
/// A constant for the value of \f$\tau\f$.
static constexpr auto TAU = static_cast<fp>(
    6.283185307179586476925286766559005768394338798750211641950L);
/// A constant for the value of \f$e\f$.
static constexpr auto E = static_cast<fp>(
    2.718281828459045235360287471352662497757247093699959574967L);

/// Supported file formats
enum class Format : uint8_t {
  /**
   * @brief OpenQASM 2.0 format
   * @see https://arxiv.org/abs/1707.03429
   */
  OpenQASM2,
  /**
   * @brief OpenQASM 3 format
   * @see https://openqasm.com/index.html
   */
  OpenQASM3
};

/**
 * @brief Combine two 64bit hashes into one 64bit hash
 * @details Combines two 64bit hashes into one 64bit hash based on
 * boost::hash_combine (https://www.boost.org/LICENSE_1_0.txt)
 * @param lhs The first hash
 * @param rhs The second hash
 * @returns The combined hash
 */
[[nodiscard]] constexpr std::size_t
combineHash(const std::size_t lhs, const std::size_t rhs) noexcept {
  return lhs ^ (rhs + 0x9e3779b97f4a7c15ULL + (lhs << 6) + (lhs >> 2));
}

/**
 * @brief Extend a 64bit hash with a 64bit integer
 * @param hash The hash to extend
 * @param with The integer to extend the hash with
 * @return The combined hash
 */
constexpr void hashCombine(std::size_t& hash, const std::size_t with) noexcept {
  hash = combineHash(hash, with);
}

/**
 * @brief Function used to mark unreachable code
 * @details Uses compiler specific extensions if possible. Even if no extension
 * is used, undefined behavior is still raised by an empty function body and the
 * noreturn attribute.
 */
[[noreturn]] inline void unreachable() {
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#endif
}
} // namespace qc
