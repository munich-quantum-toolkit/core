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
    QCXOpTest, QCTest,
    testing::Values(
        QCTestCase{"X", x, x},
        QCTestCase{"SingleControlledX", singleControlledX, singleControlledX},
        QCTestCase{"MultipleControlledX", multipleControlledX,
                   multipleControlledX},
        QCTestCase{"NestedControlledX", nestedControlledX, multipleControlledX},
        QCTestCase{"TrivialControlledX", trivialControlledX, x},
        QCTestCase{"InverseX", inverseX, x},
        QCTestCase{"InverseMultipleControlledX", inverseMultipleControlledX,
                   multipleControlledX}),
    printTestName);
