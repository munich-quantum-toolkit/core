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
    QCRZXOpTest, QCTest,
    testing::Values(
        QCTestCase{"RZX", rzx, rzx},
        QCTestCase{"SingleControlledRZX", singleControlledRzx,
                   singleControlledRzx},
        QCTestCase{"MultipleControlledRZX", multipleControlledRzx,
                   multipleControlledRzx},
        QCTestCase{"NestedControlledRZX", nestedControlledRzx,
                   multipleControlledRzx},
        QCTestCase{"TrivialControlledRZX", trivialControlledRzx, rzx},
        QCTestCase{"InverseRZX", inverseRzx, rzx},
        QCTestCase{"InverseMultipleControlledRZX", inverseMultipleControlledRzx,
                   multipleControlledRzx}),
    printTestName);
