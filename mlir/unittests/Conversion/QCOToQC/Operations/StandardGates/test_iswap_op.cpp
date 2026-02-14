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
    QCOiSWAPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"iSWAP", MQT_NAMED_BUILDER(qco::iswap),
                        MQT_NAMED_BUILDER(qc::iswap)},
        QCOToQCTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::singleControlledIswap),
                        MQT_NAMED_BUILDER(qc::singleControlledIswap)},
        QCOToQCTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qco::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap)}));
