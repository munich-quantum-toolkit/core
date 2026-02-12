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
    QCPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"P", qc::p, qco::p},
        QCToQCOTestCase{"SingleControlledP", qc::singleControlledP,
                        qco::singleControlledP},
        QCToQCOTestCase{"MultipleControlledP", qc::multipleControlledP,
                        qco::multipleControlledP},
        QCToQCOTestCase{"NestedControlledP", qc::nestedControlledP,
                        qco::multipleControlledP},
        QCToQCOTestCase{"TrivialControlledP", qc::trivialControlledP, qco::p},
        QCToQCOTestCase{"InverseP", qc::inverseP, qco::p},
        QCToQCOTestCase{"InverseMultipleControlledP",
                        qc::inverseMultipleControlledP,
                        qco::multipleControlledP}),
    printTestName);
