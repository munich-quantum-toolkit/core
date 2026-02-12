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
    QCOROpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"R", qco::r, qc::r},
        QCOToQCTestCase{"SingleControlledR", qco::singleControlledR,
                        qc::singleControlledR},
        QCOToQCTestCase{"MultipleControlledR", qco::multipleControlledR,
                        qc::multipleControlledR},
        QCOToQCTestCase{"NestedControlledR", qco::nestedControlledR,
                        qc::multipleControlledR},
        QCOToQCTestCase{"TrivialControlledR", qco::trivialControlledR, qc::r},
        QCOToQCTestCase{"InverseR", qco::inverseR, qc::r},
        QCOToQCTestCase{"InverseMultipleControlledR",
                        qco::inverseMultipleControlledR,
                        qc::multipleControlledR}),
    printTestName);
