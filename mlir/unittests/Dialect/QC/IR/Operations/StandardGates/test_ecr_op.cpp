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
    QCECROpTest, QCTest,
    testing::Values(
        QCTestCase{"ECR", ecr, ecr},
        QCTestCase{"SingleControlledECR", singleControlledEcr,
                   singleControlledEcr},
        QCTestCase{"MultipleControlledECR", multipleControlledEcr,
                   multipleControlledEcr},
        QCTestCase{"NestedControlledECR", nestedControlledEcr,
                   multipleControlledEcr},
        QCTestCase{"TrivialControlledECR", trivialControlledEcr, ecr},
        QCTestCase{"InverseECR", inverseEcr, inverseEcr},
        QCTestCase{"InverseMultipleControlledECR", inverseMultipleControlledEcr,
                   inverseMultipleControlledEcr}),
    printTestName);
