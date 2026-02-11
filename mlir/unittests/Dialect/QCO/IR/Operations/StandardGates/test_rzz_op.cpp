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
    QCORZZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RZZ", rzz, rzz},
        QCOTestCase{"SingleControlledRZZ", singleControlledRzz,
                    singleControlledRzz},
        QCOTestCase{"MultipleControlledRZZ", multipleControlledRzz,
                    multipleControlledRzz},
        QCOTestCase{"NestedControlledRZZ", nestedControlledRzz,
                    multipleControlledRzz},
        QCOTestCase{"TrivialControlledRZZ", trivialControlledRzz, rzz},
        QCOTestCase{"InverseRZZ", inverseRzz, rzz},
        QCOTestCase{"InverseMultipleControlledRZZ",
                    inverseMultipleControlledRzz, multipleControlledRzz}),
    printTestName);
