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
    QCRYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RY", qc::ry, qco::ry},
        QCToQCOTestCase{"SingleControlledRY", qc::singleControlledRy,
                        qco::singleControlledRy},
        QCToQCOTestCase{"MultipleControlledRY", qc::multipleControlledRy,
                        qco::multipleControlledRy},
        QCToQCOTestCase{"NestedControlledRY", qc::nestedControlledRy,
                        qco::multipleControlledRy},
        QCToQCOTestCase{"TrivialControlledRY", qc::trivialControlledRy,
                        qco::ry},
        QCToQCOTestCase{"InverseRY", qc::inverseRy, qco::ry},
        QCToQCOTestCase{"InverseMultipleControlledRY",
                        qc::inverseMultipleControlledRy,
                        qco::multipleControlledRy}),
    printTestName);
