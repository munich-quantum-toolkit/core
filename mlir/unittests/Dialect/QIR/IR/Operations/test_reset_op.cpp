/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir_programs.h"
#include "test_qir_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qir;

INSTANTIATE_TEST_SUITE_P(
    QIRResetOpTest, QIRTest,
    testing::Values(
        QIRTestCase{"ResetQubitWithoutOp",
                    MQT_NAMED_BUILDER(resetQubitWithoutOp),
                    MQT_NAMED_BUILDER(resetQubitWithoutOp)},
        QIRTestCase{"ResetMultipleQubitsWithoutOp",
                    MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                    MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp)},
        QIRTestCase{"RepeatedResetWithoutOp",
                    MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                    MQT_NAMED_BUILDER(repeatedResetWithoutOp)},
        QIRTestCase{"ResetQubitAfterSingleOp",
                    MQT_NAMED_BUILDER(resetQubitAfterSingleOp),
                    MQT_NAMED_BUILDER(resetQubitAfterSingleOp)},
        QIRTestCase{"ResetMultipleQubitsAfterSingleOp",
                    MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp),
                    MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp)},
        QIRTestCase{"RepeatedResetAfterSingleOp",
                    MQT_NAMED_BUILDER(repeatedResetAfterSingleOp),
                    MQT_NAMED_BUILDER(repeatedResetAfterSingleOp)}));
