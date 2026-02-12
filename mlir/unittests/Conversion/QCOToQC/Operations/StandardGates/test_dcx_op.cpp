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
    QCODCXOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"DCX", qco::dcx, qc::dcx},
        QCOToQCTestCase{"SingleControlledDCX", qco::singleControlledDcx,
                        qc::singleControlledDcx},
        QCOToQCTestCase{"MultipleControlledDCX", qco::multipleControlledDcx,
                        qc::multipleControlledDcx},
        QCOToQCTestCase{"NestedControlledDCX", qco::nestedControlledDcx,
                        qc::multipleControlledDcx},
        QCOToQCTestCase{"TrivialControlledDCX", qco::trivialControlledDcx,
                        qc::dcx},
        QCOToQCTestCase{"InverseDCX", qco::inverseDcx, qc::dcx},
        QCOToQCTestCase{"InverseMultipleControlledDCX",
                        qco::inverseMultipleControlledDcx,
                        qc::multipleControlledDcx}),
    printTestName);
