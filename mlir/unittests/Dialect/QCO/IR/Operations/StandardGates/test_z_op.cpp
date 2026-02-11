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
    QCOZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Z", z, z},
        QCOTestCase{"SingleControlledZ", singleControlledZ, singleControlledZ},
        QCOTestCase{"MultipleControlledZ", multipleControlledZ,
                    multipleControlledZ},
        QCOTestCase{"NestedControlledZ", nestedControlledZ,
                    multipleControlledZ},
        QCOTestCase{"TrivialControlledZ", trivialControlledZ, z},
        QCOTestCase{"InverseZ", inverseZ, z},
        QCOTestCase{"InverseMultipleControlledZ", inverseMultipleControlledZ,
                    multipleControlledZ}),
    printTestName);
