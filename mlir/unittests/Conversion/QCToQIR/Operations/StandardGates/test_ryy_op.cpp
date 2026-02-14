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
    QCToQIRRYYOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                        MQT_NAMED_BUILDER(qir::ryy)},
        QCToQIRTestCase{"SingleControlledRYY",
                        MQT_NAMED_BUILDER(qc::singleControlledRyy),
                        MQT_NAMED_BUILDER(qir::singleControlledRyy)},
        QCToQIRTestCase{"MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRyy)}));
