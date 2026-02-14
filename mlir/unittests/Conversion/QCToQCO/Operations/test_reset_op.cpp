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
#include "qco_programs.h"
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCResetOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)},
        QCToQCOTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp)},
        QCToQCOTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)}));
