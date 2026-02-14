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
    QCOXXMinusYYOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qco::xxMinusYY),
                        MQT_NAMED_BUILDER(qc::xxMinusYY)},
        QCOToQCTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY)},
        QCOToQCTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY)}));
