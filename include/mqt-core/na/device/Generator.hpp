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

#include "na/device/device.pb.h"

#include <istream>
#include <ostream>
#include <string>

namespace na {
/**
 * @brief Writes a JSON schema with default values for the device configuration
 * to the specified output stream.
 * @param os is the output stream to write the JSON schema to.
 * @throws std::runtime_error if the JSON conversion fails.
 */
auto writeJSONSchema(std::ostream& os) -> void;

/**
 * @brief Writes a JSON schema with default values for the device configuration
 * to the specified path.
 * @param path The path to write the JSON schema to.
 * @throws std::runtime_error if the JSON conversion fails or the file cannot be
 * opened.
 */
auto writeJSONSchema(const std::string& path) -> void;

/**
 * @brief Parses the device configuration from an input stream.
 * @param is is the input stream containing the JSON representation of the
 * device configuration.
 * @returns The parsed device configuration as a Protobuf message.
 * @throws std::runtime_error if the JSON cannot be parsed.
 */
[[nodiscard]] auto readJSON(std::istream& is) -> Device;

/**
 * @brief Parses the device configuration from a JSON file.
 * @param path is the path to the JSON file containing the device configuration.
 * @returns The parsed device configuration as a Protobuf message.
 * @throws std::runtime_error if the JSON file does not exist, or the JSON file
 * cannot be parsed.
 */
[[nodiscard]] auto readJSON(const std::string& path) -> Device;

/**
 * @brief Writes a header file with the device configuration to the specified
 * output stream.
 * @param device is the protobuf representation of the device.
 * @param os is the output stream to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 */
auto writeHeader(const Device& device, std::ostream& os) -> void;

/**
 * @brief Writes a header file with the device configuration to the specified
 * path.
 * @param device is the protobuf representation of the device.
 * @param path is the path to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 */
auto writeHeader(const Device& device, const std::string& path) -> void;

/**
 * @brief Solves a 2D linear equation system.
 * @details The equation has the following form:
 * @code
 * x1 * i + x2 * j = x0
 * y1 * i + y2 * j = y0
 * @endcode
 * The free variables are i and j.
 * @param x1 Coefficient for x in the first equation.
 * @param x2 Coefficient for y in the first equation.
 * @param y1 Coefficient for x in the second equation.
 * @param y2 Coefficient for y in the second equation.
 * @param x0 Right-hand side of the first equation.
 * @param y0 Right-hand side of the second equation.
 * @returns A pair containing the solution (x, y).
 * @throws std::runtime_error if the system has no unique solution (determinant
 * is zero).
 */
template <typename T>
[[nodiscard]] auto solve2DLinearEquation(const T x1, const T x2, const T y1,
                                         const T y2, const T x0, const T y0)
    -> std::pair<double, double> {
  // Calculate the determinant
  const auto D = static_cast<double>(x1 * y2 - x2 * y1);
  if (D == 0) {
    throw std::runtime_error("The system of equations has no unique solution.");
  }
  // Calculate the solution
  const auto Dx = static_cast<double>(x0 * y2 - x2 * y0);
  const auto Dy = static_cast<double>(x1 * y0 - x0 * y1);
  return {Dx / D, Dy / D};
}

} // namespace na
