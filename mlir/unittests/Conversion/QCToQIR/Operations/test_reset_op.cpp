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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRResetOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"ResetQubitWithoutOp", qc::resetQubitWithoutOp,
                        qir::resetQubitWithoutOp},
        QCToQIRTestCase{"ResetMultipleQubitsWithoutOp",
                        qc::resetMultipleQubitsWithoutOp,
                        qir::resetMultipleQubitsWithoutOp},
        QCToQIRTestCase{"RepeatedResetWithoutOp", qc::repeatedResetWithoutOp,
                        qir::repeatedResetWithoutOp},
        QCToQIRTestCase{"ResetQubitAfterSingleOp", qc::resetQubitAfterSingleOp,
                        qir::resetQubitAfterSingleOp},
        QCToQIRTestCase{"ResetMultipleQubitsAfterSingleOp",
                        qc::resetMultipleQubitsAfterSingleOp,
                        qir::resetMultipleQubitsAfterSingleOp},
        QCToQIRTestCase{"RepeatedResetAfterSingleOp",
                        qc::repeatedResetAfterSingleOp,
                        qir::repeatedResetAfterSingleOp}),
    printTestName);
