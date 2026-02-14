/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCResetOpTest, QCTest,
    testing::Values(QCTestCase{"ResetQubitWithoutOp",
                               MQT_NAMED_BUILDER(resetQubitWithoutOp),
                               MQT_NAMED_BUILDER(resetQubitWithoutOp)},
                    QCTestCase{"ResetMultipleQubitsWithoutOp",
                               MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                               MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp)},
                    QCTestCase{"RepeatedResetWithoutOp",
                               MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                               MQT_NAMED_BUILDER(repeatedResetWithoutOp)},
                    QCTestCase{"ResetQubitAfterSingleOp",
                               MQT_NAMED_BUILDER(resetQubitAfterSingleOp),
                               MQT_NAMED_BUILDER(resetQubitAfterSingleOp)},
                    QCTestCase{
                        "ResetMultipleQubitsAfterSingleOp",
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp),
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp)},
                    QCTestCase{"RepeatedResetAfterSingleOp",
                               MQT_NAMED_BUILDER(repeatedResetAfterSingleOp),
                               MQT_NAMED_BUILDER(repeatedResetAfterSingleOp)}));
