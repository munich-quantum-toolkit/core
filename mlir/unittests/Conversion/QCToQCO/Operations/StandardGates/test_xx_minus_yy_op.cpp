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
    QCXXMinusYYOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"XXMinusYY", qc::xxMinusYY, qco::xxMinusYY},
                    QCToQCOTestCase{"SingleControlledXXMinusYY",
                                    qc::singleControlledXxMinusYY,
                                    qco::singleControlledXxMinusYY},
                    QCToQCOTestCase{"MultipleControlledXXMinusYY",
                                    qc::multipleControlledXxMinusYY,
                                    qco::multipleControlledXxMinusYY},
                    QCToQCOTestCase{"NestedControlledXXMinusYY",
                                    qc::nestedControlledXxMinusYY,
                                    qco::multipleControlledXxMinusYY},
                    QCToQCOTestCase{"TrivialControlledXXMinusYY",
                                    qc::trivialControlledXxMinusYY,
                                    qco::xxMinusYY},
                    QCToQCOTestCase{"InverseXXMinusYY", qc::inverseXxMinusYY,
                                    qco::xxMinusYY},
                    QCToQCOTestCase{"InverseMultipleControlledXXMinusYY",
                                    qc::inverseMultipleControlledXxMinusYY,
                                    qco::multipleControlledXxMinusYY}),
    printTestName);
