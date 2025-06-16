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
#include <vector>

namespace na {
struct Library {
  std::string prefix;
  std::string path;
};

/**
 * @brief Initializes the QDMI driver.
 * @details This function initializes the QDMI devices by allocating and
 * initializing a single device session for each device.
 * @param additionalLibraries A vector of additional libraries to load.
 * @returns void
 */
auto initialize(const std::vector<Library>& additionalLibraries = {}) -> void;

/**
 * @brief Finalizes the QDMI driver.
 * @details This function finalizes the QDMI driver by freeing the device
 * sessions and closing the devices.
 * @returns void
 */
auto finalize() -> void;
} // namespace na
