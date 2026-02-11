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
    QCORXXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RXX", rxx, rxx},
        QCOTestCase{"SingleControlledRXX", singleControlledRxx,
                    singleControlledRxx},
        QCOTestCase{"MultipleControlledRXX", multipleControlledRxx,
                    multipleControlledRxx},
        QCOTestCase{"NestedControlledRXX", nestedControlledRxx,
                    multipleControlledRxx},
        QCOTestCase{"TrivialControlledRXX", trivialControlledRxx, rxx},
        QCOTestCase{"InverseRXX", inverseRxx, rxx},
        QCOTestCase{"InverseMultipleControlledRXX",
                    inverseMultipleControlledRxx, multipleControlledRxx}),
    printTestName);
