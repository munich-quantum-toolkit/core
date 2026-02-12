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
    QCSWAPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SWAP", qc::swap, qco::swap},
        QCToQCOTestCase{"SingleControlledSWAP", qc::singleControlledSwap,
                        qco::singleControlledSwap},
        QCToQCOTestCase{"MultipleControlledSWAP", qc::multipleControlledSwap,
                        qco::multipleControlledSwap},
        QCToQCOTestCase{"NestedControlledSWAP", qc::nestedControlledSwap,
                        qco::multipleControlledSwap},
        QCToQCOTestCase{"TrivialControlledSWAP", qc::trivialControlledSwap,
                        qco::swap},
        QCToQCOTestCase{"InverseSWAP", qc::inverseSwap, qco::swap},
        QCToQCOTestCase{"InverseMultipleControlledSWAP",
                        qc::inverseMultipleControlledSwap,
                        qco::multipleControlledSwap}),
    printTestName);
