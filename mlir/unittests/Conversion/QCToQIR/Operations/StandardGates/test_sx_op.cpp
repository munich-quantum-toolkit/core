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
    QCToQIRSXOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"SX", qc::sx, qir::sx},
                    QCToQIRTestCase{"SingleControlledSX",
                                    qc::singleControlledSx,
                                    qir::singleControlledSx},
                    QCToQIRTestCase{"MultipleControlledSX",
                                    qc::multipleControlledSx,
                                    qir::multipleControlledSx}),
    printTestName);
