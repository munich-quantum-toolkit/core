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
    QCOXXPlusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qco::xxPlusYY),
                        MQT_NAMED_BUILDER(qc::xxPlusYY)},
        QCOToQCTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY)},
        QCOToQCTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY)}));
