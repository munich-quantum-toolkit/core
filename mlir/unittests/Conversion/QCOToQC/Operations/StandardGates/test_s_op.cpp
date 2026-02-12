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
    QCOSOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"S", qco::s, qc::s},
        QCOToQCTestCase{"SingleControlledS", qco::singleControlledS,
                        qc::singleControlledS},
        QCOToQCTestCase{"MultipleControlledS", qco::multipleControlledS,
                        qc::multipleControlledS},
        QCOToQCTestCase{"NestedControlledS", qco::nestedControlledS,
                        qc::multipleControlledS},
        QCOToQCTestCase{"TrivialControlledS", qco::trivialControlledS, qc::s},
        QCOToQCTestCase{"InverseS", qco::inverseS, qc::sdg},
        QCOToQCTestCase{"InverseMultipleControlledS",
                        qco::inverseMultipleControlledS,
                        qc::multipleControlledSdg}),
    printTestName);
