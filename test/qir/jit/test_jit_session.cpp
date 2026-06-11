/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/jit/Session.hpp"
#include "qir/runtime/Runtime.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>

namespace {

class JitSessionExecutionMode : public testing::Test {
protected:
  std::ostringstream sink;

  void SetUp() override {
    auto& runtime = qir::Runtime::getInstance();
    runtime.reset();
    runtime.setOstream(sink);
  }
  void TearDown() override { qir::Runtime::getInstance().resetOstream(); }
};

TEST_F(JitSessionExecutionMode, SamplingRecordsOutputs) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  // qir::Execution::Sampling is the default Execution mode
  const qir::JitSession session(path.string());
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(qir::Runtime::getInstance().getRecordedOutputs().empty());
}

TEST_F(JitSessionExecutionMode, StateExtractionLeavesNoRecordedOutputs) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  const qir::JitSession session(path.string(), qir::Execution::StateExtraction);
  ASSERT_EQ(session.run(), 0);
  EXPECT_TRUE(qir::Runtime::getInstance().getRecordedOutputs().empty());
}

} // namespace
