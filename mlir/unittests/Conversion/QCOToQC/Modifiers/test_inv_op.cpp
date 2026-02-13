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
    QCOInvOpTest, QCOToQCTest,
    testing::Values(
        // ECR cannot be inverted with current canonicalization
        QCOToQCTestCase{"InverseECR", qco::inverseEcr, qc::inverseEcr},
        QCOToQCTestCase{"InverseMultipleControlledECR",
                        qco::inverseMultipleControlledEcr,
                        qc::inverseMultipleControlledEcr},
        // ECR cannot be inverted with current canonicalization
        QCOToQCTestCase{"InverseiSWAP", qco::inverseIswap, qc::inverseIswap},
        QCOToQCTestCase{"InverseMultipleControllediSWAP",
                        qco::inverseMultipleControlledIswap,
                        qc::inverseMultipleControlledIswap},
        // Inverse DCX is not canonicalized in QCO
        QCOToQCTestCase{"InverseDCX", qco::inverseDcx, qc::dcx},
        QCOToQCTestCase{"InverseMultipleControlledDCX",
                        qco::inverseMultipleControlledDcx,
                        qc::multipleControlledDcx}),
    printTestName);
