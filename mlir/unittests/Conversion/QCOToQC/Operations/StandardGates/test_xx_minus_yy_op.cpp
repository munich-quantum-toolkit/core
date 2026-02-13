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
    QCOXXMinusYYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"XXMinusYY", qco::xxMinusYY, qc::xxMinusYY},
                    QCOToQCTestCase{"SingleControlledXXMinusYY",
                                    qco::singleControlledXxMinusYY,
                                    qc::singleControlledXxMinusYY},
                    QCOToQCTestCase{"MultipleControlledXXMinusYY",
                                    qco::multipleControlledXxMinusYY,
                                    qc::multipleControlledXxMinusYY}),
    printTestName);
