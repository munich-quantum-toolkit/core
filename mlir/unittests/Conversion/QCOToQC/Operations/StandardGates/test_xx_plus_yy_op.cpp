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
    QCOXXPlusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXPlusYY", qco::xxPlusYY, qc::xxPlusYY},
        QCOToQCTestCase{"SingleControlledXXPlusYY",
                        qco::singleControlledXxPlusYY,
                        qc::singleControlledXxPlusYY},
        QCOToQCTestCase{"MultipleControlledXXPlusYY",
                        qco::multipleControlledXxPlusYY,
                        qc::multipleControlledXxPlusYY},
        QCOToQCTestCase{"NestedControlledXXPlusYY",
                        qco::nestedControlledXxPlusYY,
                        qc::multipleControlledXxPlusYY},
        QCOToQCTestCase{"TrivialControlledXXPlusYY",
                        qco::trivialControlledXxPlusYY, qc::xxPlusYY},
        QCOToQCTestCase{"InverseXXPlusYY", qco::inverseXxPlusYY, qc::xxPlusYY},
        QCOToQCTestCase{"InverseMultipleControlledXXPlusYY",
                        qco::inverseMultipleControlledXxPlusYY,
                        qc::multipleControlledXxPlusYY}),
    printTestName);
