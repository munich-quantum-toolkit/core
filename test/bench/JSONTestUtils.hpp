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

#include <string_view>

namespace mqt::bench::test {

template <class Action>
void expectInvalidJSON(const Action& operation,
                       const std::string_view diagnostic) {
  const auto error = ::mqt::test::diagnostic(operation);
  ASSERT_TRUE(error);
  EXPECT_EQ(error->category, ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_NE(std::string_view(error->message).find(diagnostic),
            std::string_view::npos)
      << error->message;
}

} // namespace mqt::bench::test
