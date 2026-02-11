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

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCORZXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RZX", rzx, rzx},
        QCOTestCase{"SingleControlledRZX", singleControlledRzx,
                    singleControlledRzx},
        QCOTestCase{"MultipleControlledRZX", multipleControlledRzx,
                    multipleControlledRzx},
        QCOTestCase{"NestedControlledRZX", nestedControlledRzx,
                    multipleControlledRzx},
        QCOTestCase{"TrivialControlledRZX", trivialControlledRzx, rzx},
        QCOTestCase{"InverseRZX", inverseRzx, rzx},
        QCOTestCase{"InverseMultipleControlledRZX",
                    inverseMultipleControlledRzx, multipleControlledRzx}),
    printTestName);
