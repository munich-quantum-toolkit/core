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
    QCToQIRSOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                                    MQT_NAMED_BUILDER(qir::s)},
                    QCToQIRTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qc::singleControlledS),
                                    MQT_NAMED_BUILDER(qir::singleControlledS)},
                    QCToQIRTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qc::multipleControlledS),
                        MQT_NAMED_BUILDER(qir::multipleControlledS)}));
