/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file DDDefinitions.hpp
 * @brief Fundamental decision-diagram types, constants, and helper functions.
 */

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <numbers>
#include <set>
#include <sstream>
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

/// Qubit indices targeted by an operation.
using Targets = std::vector<Qubit>;

/// A control qubit and its polarity.
struct Control {
  /// Control polarity.
  enum class Type : bool {
    /// Positive controls trigger on \f$\ket{1}\f$.
    Pos = true,
    /// Negative controls trigger on \f$\ket{0}\f$.
    Neg = false
  };

  /// Control qubit index.
  Qubit qubit{};
  /// Control polarity.
  Type type = Type::Pos;

  /// Allow implicit conversion from a qubit index.
  /// NOLINTBEGIN(google-explicit-constructor)
  Control(const Qubit q = {}, const Type t = Type::Pos) : qubit(q), type(t) {}
  /// NOLINTEND(google-explicit-constructor)

  [[nodiscard]] std::string toString() const {
    std::ostringstream oss{};
    oss << "Control(qubit=" << qubit << ", type_=\""
        << (type == Type::Pos ? "Pos" : "Neg") << "\")";
    return oss.str();
  }
};

inline bool operator<(const Control& lhs, const Control& rhs) {
  return lhs.qubit < rhs.qubit ||
         (lhs.qubit == rhs.qubit && lhs.type < rhs.type);
}

inline bool operator==(const Control& lhs, const Control& rhs) {
  return lhs.qubit == rhs.qubit && lhs.type == rhs.type;
}

/// Compare controls by qubit index.
struct CompareControl {
  using is_transparent [[maybe_unused]] = void;

  bool operator()(const Control& lhs, const Control& rhs) const {
    return lhs < rhs;
  }
  bool operator()(const Qubit lhs, const Control& rhs) const {
    return lhs < rhs.qubit;
  }
  bool operator()(const Control& lhs, const Qubit rhs) const {
    return lhs.qubit < rhs;
  }
};

/// Controls sorted by qubit index and polarity.
using Controls = std::set<Control, CompareControl>;

/// Map logical qubit indices to physical qubit indices.
using Permutation = std::map<Qubit, Qubit>;

/**
 * @brief Floating point type to use for computations
 * @note Adjusting the precision might lead to unexpected results.
 */
using fp = double;
static_assert(std::is_floating_point_v<fp>,
              "fp should be a floating point type (float or double)");

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

static constexpr auto SQRT2_2 = static_cast<fp>(
    0.707106781186547524400844362104849039284835937688474036588L);
static constexpr fp PI = std::numbers::pi;
static constexpr auto PI_2 = PI / 2;
static constexpr fp PI_4 = PI / 4;

/// Combine two hashes with the Boost hash-combine formula.
[[nodiscard]] constexpr std::size_t
combineHash(const std::size_t lhs, const std::size_t rhs) noexcept {
  return lhs ^ (rhs + 0x9e3779b97f4a7c15ULL + (lhs << 6) + (lhs >> 2));
}

/// Add an integer to a hash.
constexpr void hashCombine(std::size_t& hash, const std::size_t with) noexcept {
  hash = combineHash(hash, with);
}

static constexpr std::uint64_t SERIALIZATION_VERSION = 1;

struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t>& p) const noexcept {
    return combineHash(p.first, p.second);
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
/// Dimension of a three-qubit gate matrix (`2^3`).
static constexpr std::uint8_t THREE_QUBIT_GATE_DIM = 8;
using ThreeQubitGateMatrix =
    std::array<std::array<std::complex<fp>, THREE_QUBIT_GATE_DIM>,
               THREE_QUBIT_GATE_DIM>;

/**
 * @brief Converts a decimal number to a binary string (big endian)
 * @param value The decimal number to convert
 * @param nbits The number of bits to use for the binary representation
 * @return The binary representation of the decimal number
 */
[[nodiscard, maybe_unused]] static std::string
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
[[nodiscard, maybe_unused]] static std::size_t ulpDistance(fp a, fp b) {
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

struct vNode;
struct mNode;

template <typename T>
concept IsVector = std::is_same_v<T, vNode>;
template <typename T>
concept IsMatrix = std::is_same_v<T, mNode>;

} // namespace dd

template <> struct std::hash<dd::Control> {
  std::size_t operator()(const dd::Control& control) const noexcept {
    return std::hash<dd::Qubit>{}(control.qubit) ^
           std::hash<dd::Control::Type>{}(control.type);
  }
};
