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

#define PREFIX MQT_NA
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define PREFIXED(name) CAT(PREFIX, _##name)

namespace na {
struct Library {
  std::string prefix;
  std::string path;
};

/**
 * @brief Initializes the QDMI driver.
 * @details This function initializes the only QDMI device by allocating and
 * initializing a device session.
 * @param additionalLibraries A vector of additional libraries to load.
 * @returns void
 */
auto initialize(const std::vector<Library>& additionalLibraries = {}) -> void;

/**
 * @brief Finalizes the QDMI driver.
 * @details This function finalizes the QDMI driver by freeing the device
 * session and closing the device.
 * @returns void
 */
auto finalize() -> void;
} // namespace na
