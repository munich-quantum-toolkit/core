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
    QCORYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RY", qco::ry, qc::ry},
        QCOToQCTestCase{"SingleControlledRY", qco::singleControlledRy,
                        qc::singleControlledRy},
        QCOToQCTestCase{"MultipleControlledRY", qco::multipleControlledRy,
                        qc::multipleControlledRy},
        QCOToQCTestCase{"NestedControlledRY", qco::nestedControlledRy,
                        qc::multipleControlledRy},
        QCOToQCTestCase{"TrivialControlledRY", qco::trivialControlledRy,
                        qc::ry},
        QCOToQCTestCase{"InverseRY", qco::inverseRy, qc::ry},
        QCOToQCTestCase{"InverseMultipleControlledRY",
                        qco::inverseMultipleControlledRy,
                        qc::multipleControlledRy}),
    printTestName);
