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

#include <string>

namespace na {
/**
 * @brief Writes a JSON schema with default values for the device configuration
 * to the specified path.
 * @param path The path to write the JSON schema to.
 * @throws std::runtime_error if the JSON conversion fails or the file cannot be
 * opened.
 */
auto writeJsonSchema(const std::string& path) -> void;

/**
 * @brief Writes a header file with the device configuration to the specified
 * path.
 * @param json is the path to the JSON file containing the device configuration.
 * @param path is the path to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 */
auto writeHeaderFile(const std::string& json, const std::string& path) -> void;

} // namespace na
