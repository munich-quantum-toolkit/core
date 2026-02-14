/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SWAP", MQT_NAMED_BUILDER(qco::swap),
                        MQT_NAMED_BUILDER(qc::swap)},
        QCOToQCTestCase{"SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qco::singleControlledSwap),
                        MQT_NAMED_BUILDER(qc::singleControlledSwap)},
        QCOToQCTestCase{"MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qco::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap)}));
