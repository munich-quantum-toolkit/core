/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qco_programs.h"
#include "test_qco_ir.h"

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"X", x, x},
        QCOTestCase{"SingleControlledX", singleControlledX, singleControlledX},
        QCOTestCase{"MultipleControlledX", multipleControlledX,
                    multipleControlledX},
        QCOTestCase{"NestedControlledX", nestedControlledX,
                    multipleControlledX},
        QCOTestCase{"TrivialControlledX", trivialControlledX, x},
        QCOTestCase{"InverseX", inverseX, x},
        QCOTestCase{"InverseMultipleControlledX", inverseMultipleControlledX,
                    multipleControlledX}),
    printTestName);
