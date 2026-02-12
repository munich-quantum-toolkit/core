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
    QCUOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"U", qc::u, qco::u},
        QCToQCOTestCase{"SingleControlledU", qc::singleControlledU,
                        qco::singleControlledU},
        QCToQCOTestCase{"MultipleControlledU", qc::multipleControlledU,
                        qco::multipleControlledU},
        QCToQCOTestCase{"NestedControlledU", qc::nestedControlledU,
                        qco::multipleControlledU},
        QCToQCOTestCase{"TrivialControlledU", qc::trivialControlledU, qco::u},
        QCToQCOTestCase{"InverseU", qc::inverseU, qco::u},
        QCToQCOTestCase{"InverseMultipleControlledU",
                        qc::inverseMultipleControlledU,
                        qco::multipleControlledU}),
    printTestName);
