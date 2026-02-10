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
    QCHOpTest, QCTest,
    testing::Values(
        QCTestCase{"H", h, h},
        QCTestCase{"SingleControlledH", singleControlledH, singleControlledH},
        QCTestCase{"MultipleControlledH", multipleControlledH,
                   multipleControlledH},
        QCTestCase{"NestedControlledH", nestedControlledH, multipleControlledH},
        QCTestCase{"TrivialControlledH", trivialControlledH, h},
        QCTestCase{"InverseH", inverseH, h},
        QCTestCase{"InverseMultipleControlledH", inverseMultipleControlledH,
                   multipleControlledH}),
    printTestName);
