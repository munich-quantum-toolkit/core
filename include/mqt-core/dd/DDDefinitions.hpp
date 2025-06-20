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

#include "ir/Definitions.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dd {
/**
 * @brief Integer type used for indexing qubits
 * @details `std::uint16_t` can address up to 65536 qubits as [0, ..., 65535].
 * @note If you need even more qubits, this can be increased to `std::uint32_t`.
 * Beware of the increased memory footprint of matrix nodes.
 */
using Qubit = std::uint16_t;

/**
 * @brief Floating point type to use for computations
 * @note Adjusting the precision might lead to unexpected results.
 */
using fp = double;
static_assert(
    std::is_floating_point_v<fp>,
    "fp should be a floating point type (float, *double*, long double)");

// logic radix
static constexpr std::uint8_t RADIX = 2;
// max no. of edges = RADIX^2
static constexpr std::uint8_t NEDGE = RADIX * RADIX;

enum class BasisStates : std::uint8_t {
  zero,  // NOLINT(readability-identifier-naming)
  one,   // NOLINT(readability-identifier-naming)
  plus,  // NOLINT(readability-identifier-naming)
  minus, // NOLINT(readability-identifier-naming)
  right, // NOLINT(readability-identifier-naming)
  left   // NOLINT(readability-identifier-naming)
};

static constexpr fp SQRT2_2 = static_cast<fp>(
    0.707106781186547524400844362104849039284835937688474036588L);
static constexpr fp PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);
static constexpr fp PI_2 = static_cast<fp>(
    1.570796326794896619231321691639751442098584699687552910487L);
static constexpr fp PI_4 = static_cast<fp>(
    0.785398163397448309615660845819875721049292349843776455243L);

static constexpr std::uint64_t SERIALIZATION_VERSION = 1;

struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
    return qc::combineHash(p.first, p.second);
  }
};

using CVec = std::vector<std::complex<fp>>;
using SparseCVec = std::unordered_map<std::size_t, std::complex<fp>>;
using SparsePVec = std::unordered_map<std::size_t, fp>;
using SparsePVecStrKeys = std::unordered_map<std::string, fp>;
using CMat = std::vector<CVec>;
using SparseCMat = std::unordered_map<std::pair<std::size_t, std::size_t>,
                                      std::complex<fp>, PairHash>;

using GateMatrix = std::array<std::complex<fp>, NEDGE>;
using TwoQubitGateMatrix =
    std::array<std::array<std::complex<fp>, NEDGE>, NEDGE>;

/**
 * @brief Converts a decimal number to a binary string (big endian)
 * @param value The decimal number to convert
 * @param nbits The number of bits to use for the binary representation
 * @return The binary representation of the decimal number
 */
[[nodiscard]] static inline std::string
intToBinaryString(const std::size_t value, const std::size_t nbits) {
  std::string binary(nbits, '0');
  for (std::size_t j = 0; j < nbits; ++j) {
    if ((value & (1ULL << j)) != 0U) {
      binary[nbits - 1 - j] = '1';
    }
  }
  return binary;
}

// calculates the Units in Last Place (ULP) distance of two floating point
// numbers
[[maybe_unused]] static std::size_t ulpDistance(fp a, fp b) {
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  if (a == b) {
    return 0;
  }

  std::size_t ulps = 1;
  fp nextFP = std::nextafter(a, b);
  // NOLINTNEXTLINE(clang-diagnostic-float-equal)
  while (nextFP != b) {
    ulps++;
    nextFP = std::nextafter(nextFP, b);
  }
  return ulps;
}

/**
 * @brief 64bit mixing hash (from MurmurHash3)
 * @details Hash function for 64bit integers adapted from MurmurHash3
 * @param k the number to hash
 * @returns the hash value
 * @see https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */
[[nodiscard]] constexpr std::size_t murmur64(std::size_t k) noexcept {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

} // namespace dd
