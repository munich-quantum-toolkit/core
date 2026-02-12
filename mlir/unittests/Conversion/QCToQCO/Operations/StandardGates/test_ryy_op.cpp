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
    QCRYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RYY", qc::ryy, qco::ryy},
        QCToQCOTestCase{"SingleControlledRYY", qc::singleControlledRyy,
                        qco::singleControlledRyy},
        QCToQCOTestCase{"MultipleControlledRYY", qc::multipleControlledRyy,
                        qco::multipleControlledRyy},
        QCToQCOTestCase{"NestedControlledRYY", qc::nestedControlledRyy,
                        qco::multipleControlledRyy},
        QCToQCOTestCase{"TrivialControlledRYY", qc::trivialControlledRyy,
                        qco::ryy},
        QCToQCOTestCase{"InverseRYY", qc::inverseRyy, qco::ryy},
        QCToQCOTestCase{"InverseMultipleControlledRYY",
                        qc::inverseMultipleControlledRyy,
                        qco::multipleControlledRyy}),
    printTestName);
