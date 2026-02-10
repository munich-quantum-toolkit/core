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

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOTest,
    testing::Values(QCOTestCase{"ECR", ecr, ecr},
                    QCOTestCase{"SingleControlledECR", singleControlledEcr,
                                singleControlledEcr},
                    QCOTestCase{"MultipleControlledECR", multipleControlledEcr,
                                multipleControlledEcr},
                    QCOTestCase{"NestedControlledECR", nestedControlledEcr,
                                multipleControlledEcr},
                    QCOTestCase{"TrivialControlledECR", trivialControlledEcr,
                                ecr},
                    QCOTestCase{"InverseECR", inverseEcr, inverseEcr},
                    QCOTestCase{"InverseMultipleControlledECR",
                                inverseMultipleControlledEcr,
                                inverseMultipleControlledEcr}),
    printTestName);
