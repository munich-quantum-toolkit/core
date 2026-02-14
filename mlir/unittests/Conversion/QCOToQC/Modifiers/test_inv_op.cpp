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
        QCOToQCTestCase{"InverseECR", MQT_NAMED_BUILDER(qco::inverseEcr),
                        MQT_NAMED_BUILDER(qc::inverseEcr)},
        QCOToQCTestCase{"InverseMultipleControlledECR",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledEcr),
                        MQT_NAMED_BUILDER(qc::inverseMultipleControlledEcr)},
        // ECR cannot be inverted with current canonicalization
        QCOToQCTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(qco::inverseIswap),
                        MQT_NAMED_BUILDER(qc::inverseIswap)},
        QCOToQCTestCase{"InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap)},
        // Inverse DCX is not canonicalized in QCO
        QCOToQCTestCase{"InverseDCX", MQT_NAMED_BUILDER(qco::inverseDcx),
                        MQT_NAMED_BUILDER(qc::dcx)},
        QCOToQCTestCase{"InverseMultipleControlledDCX",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledDcx),
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx)}));
