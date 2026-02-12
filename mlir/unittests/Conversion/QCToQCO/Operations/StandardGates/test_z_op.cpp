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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCZOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Z", qc::z, qco::z},
        QCToQCOTestCase{"SingleControlledZ", qc::singleControlledZ,
                        qco::singleControlledZ},
        QCToQCOTestCase{"MultipleControlledZ", qc::multipleControlledZ,
                        qco::multipleControlledZ},
        QCToQCOTestCase{"NestedControlledZ", qc::nestedControlledZ,
                        qco::multipleControlledZ},
        QCToQCOTestCase{"TrivialControlledZ", qc::trivialControlledZ, qco::z},
        QCToQCOTestCase{"InverseZ", qc::inverseZ, qco::z},
        QCToQCOTestCase{"InverseMultipleControlledZ",
                        qc::inverseMultipleControlledZ,
                        qco::multipleControlledZ}),
    printTestName);
