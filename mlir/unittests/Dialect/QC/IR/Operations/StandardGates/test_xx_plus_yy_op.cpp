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
    QCXXPlusYYOpTest, QCTest,
    testing::Values(
        QCTestCase{"XXPlusYY", xxPlusYY, xxPlusYY},
        QCTestCase{"SingleControlledXXPlusYY", singleControlledXxPlusYY,
                   singleControlledXxPlusYY},
        QCTestCase{"MultipleControlledXXPlusYY", multipleControlledXxPlusYY,
                   multipleControlledXxPlusYY},
        QCTestCase{"NestedControlledXXPlusYY", nestedControlledXxPlusYY,
                   multipleControlledXxPlusYY},
        QCTestCase{"TrivialControlledXXPlusYY", trivialControlledXxPlusYY,
                   xxPlusYY},
        QCTestCase{"InverseXXPlusYY", inverseXxPlusYY, xxPlusYY},
        QCTestCase{"InverseMultipleControlledXXPlusYY",
                   inverseMultipleControlledXxPlusYY,
                   multipleControlledXxPlusYY}),
    printTestName);
