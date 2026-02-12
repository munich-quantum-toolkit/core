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
    QCOXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"X", qco::x, qc::x},
        QCOToQCTestCase{"SingleControlledX", qco::singleControlledX,
                        qc::singleControlledX},
        QCOToQCTestCase{"MultipleControlledX", qco::multipleControlledX,
                        qc::multipleControlledX},
        QCOToQCTestCase{"NestedControlledX", qco::nestedControlledX,
                        qc::multipleControlledX},
        QCOToQCTestCase{"TrivialControlledX", qco::trivialControlledX, qc::x},
        QCOToQCTestCase{"InverseX", qco::inverseX, qc::x},
        QCOToQCTestCase{"InverseMultipleControlledX",
                        qco::inverseMultipleControlledX,
                        qc::multipleControlledX}),
    printTestName);
