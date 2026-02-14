/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qc_programs.h"
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QCOToQCTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp)},
        QCOToQCTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)}));
