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
    QCRYOpTest, QCTest,
    testing::Values(QCTestCase{"RY", ry, ry},
                    QCTestCase{"SingleControlledRY", singleControlledRy,
                               singleControlledRy},
                    QCTestCase{"MultipleControlledRY", multipleControlledRy,
                               multipleControlledRy},
                    QCTestCase{"NestedControlledRY", nestedControlledRy,
                               multipleControlledRy},
                    QCTestCase{"TrivialControlledRY", trivialControlledRy, ry},
                    QCTestCase{"InverseRY", inverseRy, ry},
                    QCTestCase{"InverseMultipleControlledRY",
                               inverseMultipleControlledRy,
                               multipleControlledRy}),
    printTestName);
