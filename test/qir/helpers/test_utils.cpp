/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/helpers/test_utils.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

namespace qir_test {

std::string getProgram(const std::string_view file) {
  const std::filesystem::path path =
      std::filesystem::path(QIR_FILES_DIR) / file;
  std::ifstream ifs(path);
  EXPECT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
  return {std::istreambuf_iterator<char>{ifs}, {}};
}

} // namespace qir_test
