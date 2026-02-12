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
    QCOU2OpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"U2", qco::u2, qc::u2},
        QCOToQCTestCase{"SingleControlledU2", qco::singleControlledU2,
                        qc::singleControlledU2},
        QCOToQCTestCase{"MultipleControlledU2", qco::multipleControlledU2,
                        qc::multipleControlledU2},
        QCOToQCTestCase{"NestedControlledU2", qco::nestedControlledU2,
                        qc::multipleControlledU2},
        QCOToQCTestCase{"TrivialControlledU2", qco::trivialControlledU2,
                        qc::u2},
        QCOToQCTestCase{"InverseU2", qco::inverseU2, qc::u2},
        QCOToQCTestCase{"InverseMultipleControlledU2",
                        qco::inverseMultipleControlledU2,
                        qc::multipleControlledU2}),
    printTestName);
