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
    QCOPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"P", qco::p, qc::p},
        QCOToQCTestCase{"SingleControlledP", qco::singleControlledP,
                        qc::singleControlledP},
        QCOToQCTestCase{"MultipleControlledP", qco::multipleControlledP,
                        qc::multipleControlledP},
        QCOToQCTestCase{"NestedControlledP", qco::nestedControlledP,
                        qc::multipleControlledP},
        QCOToQCTestCase{"TrivialControlledP", qco::trivialControlledP, qc::p},
        QCOToQCTestCase{"InverseP", qco::inverseP, qc::p},
        QCOToQCTestCase{"InverseMultipleControlledP",
                        qco::inverseMultipleControlledP,
                        qc::multipleControlledP}),
    printTestName);
