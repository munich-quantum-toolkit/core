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
    QCDCXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"DCX", qc::dcx, qco::dcx},
        QCToQCOTestCase{"SingleControlledDCX", qc::singleControlledDcx,
                        qco::singleControlledDcx},
        QCToQCOTestCase{"MultipleControlledDCX", qc::multipleControlledDcx,
                        qco::multipleControlledDcx},
        QCToQCOTestCase{"NestedControlledDCX", qc::nestedControlledDcx,
                        qco::multipleControlledDcx},
        QCToQCOTestCase{"TrivialControlledDCX", qc::trivialControlledDcx,
                        qco::dcx},
        QCToQCOTestCase{"InverseDCX", qc::inverseDcx, qco::dcx},
        QCToQCOTestCase{"InverseMultipleControlledDCX",
                        qc::inverseMultipleControlledDcx,
                        qco::multipleControlledDcx}),
    printTestName);
