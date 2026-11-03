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

#include "llvm/Support/Error.h"

#include <string>
#include <string_view>
#include <system_error>

namespace mqt::bench::test {

template <class T>
void expectError(llvm::Expected<T> result, const std::errc code) {
  ASSERT_FALSE(static_cast<bool>(result)) << "Expected invalid benchmark input";
  EXPECT_EQ(llvm::errorToErrorCode(result.takeError()),
            std::make_error_code(code));
}

template <class Operation>
void expectInvalidJSON(const Operation& operation,
                       const std::string_view diagnostic) {
  auto result = operation();
  ASSERT_FALSE(static_cast<bool>(result)) << "Expected invalid JSON input";
  const auto error = llvm::toString(result.takeError());
  EXPECT_NE(error.find(diagnostic), std::string::npos) << error;
}

} // namespace mqt::bench::test
