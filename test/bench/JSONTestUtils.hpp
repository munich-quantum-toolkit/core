/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "gtest/gtest.h"

#include <functional>
#include <stdexcept>
#include <string_view>

namespace mqt::bench::test {

inline void expectInvalidJSON(const std::function<void()>& operation,
                              const std::string_view diagnostic) {
  try {
    operation();
    FAIL() << "Expected invalid JSON input";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string_view(error.what()).find(diagnostic),
              std::string_view::npos)
        << error.what();
  }
}

} // namespace mqt::bench::test
