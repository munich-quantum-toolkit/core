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
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCXXMinusYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"XXMinusYY", xxMinusYY, xxMinusYY},
        QCTestCase{"SingleControlledXXMinusYY", singleControlledXxMinusYY,
                   singleControlledXxMinusYY},
        QCTestCase{"MultipleControlledXXMinusYY", multipleControlledXxMinusYY,
                   multipleControlledXxMinusYY},
        QCTestCase{"NestedControlledXXMinusYY", nestedControlledXxMinusYY,
                   multipleControlledXxMinusYY},
        QCTestCase{"TrivialControlledXXMinusYY", trivialControlledXxMinusYY,
                   xxMinusYY},
        QCTestCase{"InverseXXMinusYY", inverseXxMinusYY, xxMinusYY},
        QCTestCase{"InverseMultipleControlledXXMinusYY",
                   inverseMultipleControlledXxMinusYY,
                   multipleControlledXxMinusYY}),
    printTestName);
