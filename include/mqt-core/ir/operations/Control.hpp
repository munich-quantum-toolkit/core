/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a type for control qubits and some related functionality.
 */

#pragma once

#include "ir/Definitions.hpp"

#include <cstddef>
#include <functional>
#include <set>
#include <sstream>
#include <string>

namespace qc {
/// Represents a control qubit as a qubit index and a control type.
struct Control {
  /// The polarity of the control.
  enum class Type : bool {
    /// Positive controls trigger on \f$\ket{1}\f$.
    Pos = true,
    /// Negative controls trigger on \f$\ket{0}\f$.
    Neg = false
  };

  /// The qubit that acts as a control.
  Qubit qubit{};
  /// The type of the control.
  Type type = Type::Pos;

  /// Get a string representation of the control.
  [[nodiscard]] std::string toString() const {
    std::ostringstream oss{};
    oss << "Control(qubit=" << qubit << ", type_=\"";
    if (type == Type::Pos) {
      oss << "Pos";
    } else {
      oss << "Neg";
    }
    oss << "\")";
    return oss.str();
  }

  // Explicitly allow implicit conversion from `Qubit` to `Control`
  // NOLINTBEGIN(google-explicit-constructor)
  /**
   * @brief Construct a control qubit.
   * @param q The qubit that acts as a control.
   * @param t The type of the control. Defaults to `Type::Pos`.
   * @note This constructor is not `explicit` to allow implicit conversion from
   *       `Qubit` to `Control`.
   */
  Control(const Qubit q = {}, const Type t = Type::Pos) : qubit(q), type(t) {}
  // NOLINTEND(google-explicit-constructor)
};

/// Defines the order of controls based on their qubit index and type.
inline bool operator<(const Control& lhs, const Control& rhs) {
  return lhs.qubit < rhs.qubit ||
         (lhs.qubit == rhs.qubit && lhs.type < rhs.type);
}

/// operator== overload for `Control`
inline bool operator==(const Control& lhs, const Control& rhs) {
  return lhs.qubit == rhs.qubit && lhs.type == rhs.type;
}

/// operator!= overload for `Control`
inline bool operator!=(const Control& lhs, const Control& rhs) {
  return !(lhs == rhs);
}

/// Allows a set of @ref Control to be indexed by a `Qubit`
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

/// Type alias for a set of control qubits.
using Controls = std::set<Control, CompareControl>;

/**
 * @brief Inline namespace for control literals.
 * @details Use `using namespace qc::literals` to enable the literals.
 */
inline namespace literals {
// User-defined literals require unsigned long long int
// NOLINTBEGIN(google-runtime-int)

/**
 * @brief User-defined literal for positive control qubits.
 * @details This literal allows to create a positive control qubit from an
 *          unsigned integer, for example as `0_pc`.
 * @param q Index of the qubit that acts as a control.
 * @return A positive control qubit.
 */
inline Control operator""_pc(const unsigned long long int q) {
  return {static_cast<Qubit>(q)};
}

/**
 * @brief User-defined literal for negative control qubits.
 * @details This literal allows to create a negative control qubit from an
 *          unsigned integer, for example as `0_nc`.
 * @param q Index of the qubit that acts as a control.
 * @return A negative control qubit.
 */
inline Control operator""_nc(const unsigned long long int q) {
  return {static_cast<Qubit>(q), Control::Type::Neg};
}
// NOLINTEND(google-runtime-int)
} // namespace literals
} // namespace qc

/// Hash function for `Control`
template <> struct std::hash<qc::Control> {
  std::size_t operator()(const qc::Control& c) const noexcept {
    return std::hash<qc::Qubit>{}(c.qubit) ^
           std::hash<qc::Control::Type>{}(c.type);
  }
};
