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
    QCROpTest, QCTest,
    testing::Values(
        QCTestCase{"R", r, r},
        QCTestCase{"SingleControlledR", singleControlledR, singleControlledR},
        QCTestCase{"MultipleControlledR", multipleControlledR,
                   multipleControlledR},
        QCTestCase{"NestedControlledR", nestedControlledR, multipleControlledR},
        QCTestCase{"TrivialControlledR", trivialControlledR, r},
        QCTestCase{"InverseR", inverseR, r},
        QCTestCase{"InverseMultipleControlledR", inverseMultipleControlledR,
                   multipleControlledR}),
    printTestName);
