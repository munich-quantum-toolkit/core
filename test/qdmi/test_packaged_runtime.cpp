/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>

#ifndef MQT_CORE_QDMI_TEST_DRIVER_FILENAME
#error MQT_CORE_QDMI_TEST_DRIVER_FILENAME must name the packaged QDMI driver
#endif

int main(const int argc, const char* const argv[]) {
  try {
    if (argc != 1) {
      return EXIT_FAILURE;
    }
    const auto executable = std::filesystem::weakly_canonical(*argv);
    const auto driver =
        executable.parent_path() / MQT_CORE_QDMI_TEST_DRIVER_FILENAME;
    if (!std::filesystem::is_regular_file(driver)) {
      std::cerr << "Packaged QDMI driver is missing: " << driver << '\n';
      return EXIT_FAILURE;
    }
    std::filesystem::current_path(std::filesystem::temp_directory_path());
    qdmi::Session session;
    const auto devices = session.getDevices();
    if (!devices.empty()) {
      const auto id = devices.front().getId();
      const auto targeted = qdmi::default_driver::openDevice(id);
      if (targeted.getId() != id) {
        return EXIT_FAILURE;
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
