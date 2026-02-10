/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qco_programs.h"
#include "test_qco_ir.h"

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"ResetQubitWithoutOp", resetQubitWithoutOp, emptyQCO},
        QCOTestCase{"ResetMultipleQubitsWithoutOp",
                    resetMultipleQubitsWithoutOp, emptyQCO},
        QCOTestCase{"RepeatedResetWithoutOp", repeatedResetWithoutOp, emptyQCO},
        QCOTestCase{"ResetQubitAfterSingleOp", resetQubitAfterSingleOp,
                    resetQubitAfterSingleOp},
        QCOTestCase{"ResetMultipleQubitsAfterSingleOp",
                    resetMultipleQubitsAfterSingleOp,
                    resetMultipleQubitsAfterSingleOp},
        QCOTestCase{"RepeatedResetAfterSingleOp", repeatedResetAfterSingleOp,
                    resetQubitAfterSingleOp}),
    printTestName);
