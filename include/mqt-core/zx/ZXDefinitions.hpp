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

#include "Rational.hpp"
#include "ir/operations/Expression.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zx {

/**
 * @brief Enum for the different types of edges in the ZX-calculus
 * @details Simple edges are the standard edges in the ZX-calculus, while
 * Hadamard edges are a shorthand used to represent edges with Hadamard gates on
 * them.
 */
enum class EdgeType : uint8_t { Simple, Hadamard };
inline std::ostream& operator<<(std::ostream& os, const EdgeType& type) {
  switch (type) {
  case EdgeType::Simple:
    os << "Simple";
    break;
  case EdgeType::Hadamard:
    os << "Hadamard";
    break;
  }
  return os;
}

/**
 * @brief Enum for the different types of vertices in the ZX-calculus
 * @details Boundary vertices represent inputs and outputs. Otherwise, vertices
 * are either Z-vertices or X-vertices.
 */
enum class VertexType : uint8_t { Boundary, Z, X };
inline std::ostream& operator<<(std::ostream& os, const VertexType& type) {
  switch (type) {
  case VertexType::Boundary:
    os << "Boundary";
    break;
  case VertexType::Z:
    os << "Z";
    break;
  case VertexType::X:
    os << "X";
    break;
  }
  return os;
}

using Vertex = std::size_t;
using Col = std::int32_t;
using Qubit = std::int32_t;
using fp = double;

constexpr fp MAX_DENOM = 1e9; // TODO: maybe too high
constexpr fp PARAMETER_TOLERANCE = 1e-13;
constexpr fp TOLERANCE = 1e-13;
static constexpr auto PI = static_cast<fp>(
    3.141592653589793238462643383279502884197169399375105820974L);

using PiExpression = sym::Expression<double, PiRational>;

class ZXException : public std::invalid_argument {
  std::string msg;

public:
  explicit ZXException(std::string m)
      : std::invalid_argument("ZX Exception"), msg(std::move(m)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg.c_str();
  }
};

using gf2Mat = std::vector<std::vector<bool>>;
using gf2Vec = std::vector<bool>;
} // namespace zx
