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
    QCORXXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RXX", qco::rxx, qc::rxx},
        QCOToQCTestCase{"SingleControlledRXX", qco::singleControlledRxx,
                        qc::singleControlledRxx},
        QCOToQCTestCase{"MultipleControlledRXX", qco::multipleControlledRxx,
                        qc::multipleControlledRxx},
        QCOToQCTestCase{"NestedControlledRXX", qco::nestedControlledRxx,
                        qc::multipleControlledRxx},
        QCOToQCTestCase{"TrivialControlledRXX", qco::trivialControlledRxx,
                        qc::rxx},
        QCOToQCTestCase{"InverseRXX", qco::inverseRxx, qc::rxx},
        QCOToQCTestCase{"InverseMultipleControlledRXX",
                        qco::inverseMultipleControlledRxx,
                        qc::multipleControlledRxx}),
    printTestName);
