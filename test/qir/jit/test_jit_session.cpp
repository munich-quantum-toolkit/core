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
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
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

  static std::string getProgram(const std::string_view file) {
    const std::filesystem::path path =
        std::filesystem::path(QIR_FILES_DIR) / file;
    std::ifstream ifs(path);
    EXPECT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
    return {std::istreambuf_iterator<char>{ifs}, {}};
  }
};

TEST_F(JitSessionTest, LoadModuleFromMemory) {
  const auto program = getProgram("BellPairStatic.ll");
  const qir::JitSession session(program, "BellPairStatic.ll");
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(qir::Runtime::getInstance().getRecordedOutputs().empty());
}

TEST_F(JitSessionTest, SamplingRecordsOutputs) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  // qir::Execution::Sampling is the default Execution mode
  const qir::JitSession session(path.string());
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(qir::Runtime::getInstance().getRecordedOutputs().empty());
}

TEST_F(JitSessionTest, StateExtractionLeavesNoRecordedOutputs) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / "BellPairStatic.ll";
  const qir::JitSession session(path.string(), qir::Execution::StateExtraction);
  ASSERT_EQ(session.run(), 0);
  EXPECT_TRUE(qir::Runtime::getInstance().getRecordedOutputs().empty());
}

TEST(JitSessionErrors, MalformedIRThrows) {
  constexpr std::string_view ir = R"(define i32 @main() {})";
  EXPECT_THROW(qir::JitSession(ir, "MalformedIR.ll"), std::runtime_error);
}

} // namespace
