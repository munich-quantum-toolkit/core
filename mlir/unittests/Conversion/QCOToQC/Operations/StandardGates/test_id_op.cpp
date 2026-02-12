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
    QCOIDOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"Identity", qco::identity, emptyQC},
        QCOToQCTestCase{"SingleControlledIdentity",
                        qco::singleControlledIdentity, emptyQC},
        QCOToQCTestCase{"MultipleControlledIdentity",
                        qco::multipleControlledIdentity, emptyQC},
        QCOToQCTestCase{"NestedControlledIdentity",
                        qco::nestedControlledIdentity, emptyQC},
        QCOToQCTestCase{"TrivialControlledIdentity",
                        qco::trivialControlledIdentity, emptyQC},
        QCOToQCTestCase{"InverseIdentity", qco::inverseIdentity, emptyQC},
        QCOToQCTestCase{"InverseMultipleControlledIdentity",
                        qco::inverseMultipleControlledIdentity, emptyQC}),
    printTestName);
