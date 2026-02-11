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
    QCOHOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"H", h, h},
        QCOTestCase{"SingleControlledH", singleControlledH, singleControlledH},
        QCOTestCase{"MultipleControlledH", multipleControlledH,
                    multipleControlledH},
        QCOTestCase{"NestedControlledH", nestedControlledH,
                    multipleControlledH},
        QCOTestCase{"TrivialControlledH", trivialControlledH, h},
        QCOTestCase{"InverseH", inverseH, h},
        QCOTestCase{"InverseMultipleControlledH", inverseMultipleControlledH,
                    multipleControlledH}),
    printTestName);
