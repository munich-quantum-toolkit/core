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
    QCORZOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"RZ", qco::rz, qc::rz},
        QCOToQCTestCase{"SingleControlledRZ", qco::singleControlledRz,
                        qc::singleControlledRz},
        QCOToQCTestCase{"MultipleControlledRZ", qco::multipleControlledRz,
                        qc::multipleControlledRz},
        QCOToQCTestCase{"NestedControlledRZ", qco::nestedControlledRz,
                        qc::multipleControlledRz},
        QCOToQCTestCase{"TrivialControlledRZ", qco::trivialControlledRz,
                        qc::rz},
        QCOToQCTestCase{"InverseRZ", qco::inverseRz, qc::rz},
        QCOToQCTestCase{"InverseMultipleControlledRZ",
                        qco::inverseMultipleControlledRz,
                        qc::multipleControlledRz}),
    printTestName);
