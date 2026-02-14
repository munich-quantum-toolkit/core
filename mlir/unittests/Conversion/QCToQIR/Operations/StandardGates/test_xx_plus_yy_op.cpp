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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRXXPlusYYOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                        MQT_NAMED_BUILDER(qir::xxPlusYY)},
        QCToQIRTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qir::singleControlledXxPlusYY)},
        QCToQIRTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qir::multipleControlledXxPlusYY)}));
