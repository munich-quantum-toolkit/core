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
        QCTestCase{"ECR", MQT_NAMED_BUILDER(ecr), MQT_NAMED_BUILDER(ecr)},
        QCTestCase{"SingleControlledECR",
                   MQT_NAMED_BUILDER(singleControlledEcr),
                   MQT_NAMED_BUILDER(singleControlledEcr)},
        QCTestCase{"MultipleControlledECR",
                   MQT_NAMED_BUILDER(multipleControlledEcr),
                   MQT_NAMED_BUILDER(multipleControlledEcr)},
        QCTestCase{"NestedControlledECR",
                   MQT_NAMED_BUILDER(nestedControlledEcr),
                   MQT_NAMED_BUILDER(multipleControlledEcr)},
        QCTestCase{"TrivialControlledECR",
                   MQT_NAMED_BUILDER(trivialControlledEcr),
                   MQT_NAMED_BUILDER(ecr)},
        QCTestCase{"InverseECR", MQT_NAMED_BUILDER(inverseEcr),
                   MQT_NAMED_BUILDER(inverseEcr)},
        QCTestCase{"InverseMultipleControlledECR",
                   MQT_NAMED_BUILDER(inverseMultipleControlledEcr),
                   MQT_NAMED_BUILDER(inverseMultipleControlledEcr)}));
