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

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCZOpTest, QCTest,
    testing::Values(
        QCTestCase{"Z", z, z},
        QCTestCase{"SingleControlledZ", singleControlledZ, singleControlledZ},
        QCTestCase{"MultipleControlledZ", multipleControlledZ,
                   multipleControlledZ},
        QCTestCase{"NestedControlledZ", nestedControlledZ, multipleControlledZ},
        QCTestCase{"TrivialControlledZ", trivialControlledZ, z},
        QCTestCase{"InverseZ", inverseZ, z},
        QCTestCase{"InverseMultipleControlledZ", inverseMultipleControlledZ,
                   multipleControlledZ}),
    printTestName);
