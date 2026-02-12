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
    QCORYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RYY", qco::ryy, qc::ryy},
        QCOToQCTestCase{"SingleControlledRYY", qco::singleControlledRyy,
                        qc::singleControlledRyy},
        QCOToQCTestCase{"MultipleControlledRYY", qco::multipleControlledRyy,
                        qc::multipleControlledRyy},
        QCOToQCTestCase{"NestedControlledRYY", qco::nestedControlledRyy,
                        qc::multipleControlledRyy},
        QCOToQCTestCase{"TrivialControlledRYY", qco::trivialControlledRyy,
                        qc::ryy},
        QCOToQCTestCase{"InverseRYY", qco::inverseRyy, qc::ryy},
        QCOToQCTestCase{"InverseMultipleControlledRYY",
                        qco::inverseMultipleControlledRyy,
                        qc::multipleControlledRyy}),
    printTestName);
