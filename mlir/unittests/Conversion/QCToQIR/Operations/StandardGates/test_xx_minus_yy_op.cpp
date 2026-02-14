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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRXXMinusYYOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                        MQT_NAMED_BUILDER(qir::xxMinusYY)},
        QCToQIRTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qir::singleControlledXxMinusYY)},
        QCToQIRTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qir::multipleControlledXxMinusYY)}));
