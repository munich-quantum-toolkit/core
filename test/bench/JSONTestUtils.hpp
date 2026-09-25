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

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <functional>
#include <string_view>

namespace mqt::bench::test {

inline void expectInvalidJSON(const std::function<void()>& operation,
                              const std::string_view diagnostic) {
  ::mqt::test::DiagnosticCapture capture;
  operation();
  ASSERT_TRUE(capture.error);
  EXPECT_EQ(capture.error->category, ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_NE(std::string_view(capture.error->message).find(diagnostic),
            std::string_view::npos)
      << capture.error->message;
}

} // namespace mqt::bench::test
