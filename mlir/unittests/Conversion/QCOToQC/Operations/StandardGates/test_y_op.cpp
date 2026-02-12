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
    QCOYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"Y", qco::y, qc::y},
        QCOToQCTestCase{"SingleControlledY", qco::singleControlledY,
                        qc::singleControlledY},
        QCOToQCTestCase{"MultipleControlledY", qco::multipleControlledY,
                        qc::multipleControlledY},
        QCOToQCTestCase{"NestedControlledY", qco::nestedControlledY,
                        qc::multipleControlledY},
        QCOToQCTestCase{"TrivialControlledY", qco::trivialControlledY, qc::y},
        QCOToQCTestCase{"InverseY", qco::inverseY, qc::y},
        QCOToQCTestCase{"InverseMultipleControlledY",
                        qco::inverseMultipleControlledY,
                        qc::multipleControlledY}),
    printTestName);
