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
#include "qir/jit/Session.hpp"
#include "qir/runtime/Runtime.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

class JitSessionTest : public testing::Test {
protected:
  std::ostringstream sink;

  void SetUp() override {
    auto& runtime = qir::Runtime::getInstance();
    runtime.reset();
    runtime.setOstream(sink);
  }
  void TearDown() override { qir::Runtime::getInstance().resetOstream(); }
};

TEST_F(JitSessionTest, LoadModuleFromMemory) {
  const auto program = qir_test::getProgram("BellPairStatic.ll");
  const qir::JitSession session(program, "BellPairStatic.ll");
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(qir::Runtime::getInstance().getMeasurements().empty());
}

TEST_F(JitSessionTest, SamplingRecordsOutputs) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  // qir::Execution::Sampling is the default Execution mode
  const qir::JitSession session(path.string());
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(qir::Runtime::getInstance().getMeasurements().empty());
}

TEST_F(JitSessionTest, StateExtractionLeavesNoRecordedOutputs) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  const qir::JitSession session(path.string(), qir::Execution::StateExtraction);
  ASSERT_EQ(session.run(), 0);
  EXPECT_TRUE(qir::Runtime::getInstance().getMeasurements().empty());
}

TEST(JitSessionErrors, MalformedIRThrows) {
  constexpr std::string_view ir = R"(define i32 @main() {})";
  EXPECT_THROW(qir::JitSession(ir, "MalformedIR.ll"), std::runtime_error);
}

} // namespace
