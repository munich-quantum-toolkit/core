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
    QCYOpTest, QCTest,
    testing::Values(
        QCTestCase{"Y", y, y},
        QCTestCase{"SingleControlledY", singleControlledY, singleControlledY},
        QCTestCase{"MultipleControlledY", multipleControlledY,
                   multipleControlledY},
        QCTestCase{"NestedControlledY", nestedControlledY, multipleControlledY},
        QCTestCase{"TrivialControlledY", trivialControlledY, y},
        QCTestCase{"InverseY", inverseY, y},
        QCTestCase{"InverseMultipleControlledY", inverseMultipleControlledY,
                   multipleControlledY}),
    printTestName);
