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
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"ECR", qco::ecr, qc::ecr},
                    QCOToQCTestCase{"SingleControlledECR",
                                    qco::singleControlledEcr,
                                    qc::singleControlledEcr},
                    QCOToQCTestCase{"MultipleControlledECR",
                                    qco::multipleControlledEcr,
                                    qc::multipleControlledEcr}),
    printTestName);
