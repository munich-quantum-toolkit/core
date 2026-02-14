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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRResetOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"ResetQubitWithoutOp",
                        MQT_NAMED_BUILDER(qc::resetQubitWithoutOp),
                        MQT_NAMED_BUILDER(qir::resetQubitWithoutOp)},
        QCToQIRTestCase{"ResetMultipleQubitsWithoutOp",
                        MQT_NAMED_BUILDER(qc::resetMultipleQubitsWithoutOp),
                        MQT_NAMED_BUILDER(qir::resetMultipleQubitsWithoutOp)},
        QCToQIRTestCase{"RepeatedResetWithoutOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetWithoutOp),
                        MQT_NAMED_BUILDER(qir::repeatedResetWithoutOp)},
        QCToQIRTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qir::resetQubitAfterSingleOp)},
        QCToQIRTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qir::resetMultipleQubitsAfterSingleOp)},
        QCToQIRTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qir::repeatedResetAfterSingleOp)}));
