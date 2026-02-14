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
    QCToQIRXOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                                    MQT_NAMED_BUILDER(qir::x)},
                    QCToQIRTestCase{"SingleControlledX",
                                    MQT_NAMED_BUILDER(qc::singleControlledX),
                                    MQT_NAMED_BUILDER(qir::singleControlledX)},
                    QCToQIRTestCase{
                        "MultipleControlledX",
                        MQT_NAMED_BUILDER(qc::multipleControlledX),
                        MQT_NAMED_BUILDER(qir::multipleControlledX)}));
