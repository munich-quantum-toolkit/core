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
    QCRZZOpTest, QCTest,
    testing::Values(
        QCTestCase{"RZZ", rzz, rzz},
        QCTestCase{"SingleControlledRZZ", singleControlledRzz,
                   singleControlledRzz},
        QCTestCase{"MultipleControlledRZZ", multipleControlledRzz,
                   multipleControlledRzz},
        QCTestCase{"NestedControlledRZZ", nestedControlledRzz,
                   multipleControlledRzz},
        QCTestCase{"TrivialControlledRZZ", trivialControlledRzz, rzz},
        QCTestCase{"InverseRZZ", inverseRzz, rzz},
        QCTestCase{"InverseMultipleControlledRZZ", inverseMultipleControlledRzz,
                   multipleControlledRzz}),
    printTestName);
