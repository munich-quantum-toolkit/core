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

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCRYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"RYY", ryy, ryy},
        QCTestCase{"SingleControlledRYY", singleControlledRyy,
                   singleControlledRyy},
        QCTestCase{"MultipleControlledRYY", multipleControlledRyy,
                   multipleControlledRyy},
        QCTestCase{"NestedControlledRYY", nestedControlledRyy,
                   multipleControlledRyy},
        QCTestCase{"TrivialControlledRYY", trivialControlledRyy, ryy},
        QCTestCase{"InverseRYY", inverseRyy, ryy},
        QCTestCase{"InverseMultipleControlledRYY", inverseMultipleControlledRyy,
                   multipleControlledRyy}),
    printTestName);
