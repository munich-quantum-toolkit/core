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
    QCToQIRHOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                    MQT_NAMED_BUILDER(qir::h)},
                    QCToQIRTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qc::singleControlledH),
                                    MQT_NAMED_BUILDER(qir::singleControlledH)},
                    QCToQIRTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qc::multipleControlledH),
                        MQT_NAMED_BUILDER(qir::multipleControlledH)}));
