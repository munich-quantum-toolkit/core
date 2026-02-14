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
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOTest,
    testing::Values(QCOTestCase{"ResetQubitWithoutOp",
                                MQT_NAMED_BUILDER(resetQubitWithoutOp),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"ResetMultipleQubitsWithoutOp",
                                MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"RepeatedResetWithoutOp",
                                MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                                MQT_NAMED_BUILDER(emptyQCO)},
                    QCOTestCase{"ResetQubitAfterSingleOp",
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp),
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp)},
                    QCOTestCase{
                        "ResetMultipleQubitsAfterSingleOp",
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp),
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp)},
                    QCOTestCase{"RepeatedResetAfterSingleOp",
                                MQT_NAMED_BUILDER(repeatedResetAfterSingleOp),
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp)}));
