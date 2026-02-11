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
    QCOYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Y", y, y},
        QCOTestCase{"SingleControlledY", singleControlledY, singleControlledY},
        QCOTestCase{"MultipleControlledY", multipleControlledY,
                    multipleControlledY},
        QCOTestCase{"NestedControlledY", nestedControlledY,
                    multipleControlledY},
        QCOTestCase{"TrivialControlledY", trivialControlledY, y},
        QCOTestCase{"InverseY", inverseY, y},
        QCOTestCase{"InverseMultipleControlledY", inverseMultipleControlledY,
                    multipleControlledY}),
    printTestName);
