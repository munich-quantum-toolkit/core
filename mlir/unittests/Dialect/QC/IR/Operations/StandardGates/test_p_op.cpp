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
    QCPOpTest, QCTest,
    testing::Values(
        QCTestCase{"P", p, p},
        QCTestCase{"SingleControlledP", singleControlledP, singleControlledP},
        QCTestCase{"MultipleControlledP", multipleControlledP,
                   multipleControlledP},
        QCTestCase{"NestedControlledP", nestedControlledP, multipleControlledP},
        QCTestCase{"TrivialControlledP", trivialControlledP, p},
        QCTestCase{"InverseP", inverseP, p},
        QCTestCase{"InverseMultipleControlledP", inverseMultipleControlledP,
                   multipleControlledP}),
    printTestName);
