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
#include "qco_programs.h"
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCECROpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"ECR", qc::ecr, qco::ecr},
                    QCToQCOTestCase{"SingleControlledECR",
                                    qc::singleControlledEcr,
                                    qco::singleControlledEcr},
                    QCToQCOTestCase{"MultipleControlledECR",
                                    qc::multipleControlledEcr,
                                    qco::multipleControlledEcr}),
    printTestName);
