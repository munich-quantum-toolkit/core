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
    QCToQIRRYOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                                    MQT_NAMED_BUILDER(qir::ry)},
                    QCToQIRTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qc::singleControlledRy),
                                    MQT_NAMED_BUILDER(qir::singleControlledRy)},
                    QCToQIRTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRy),
                        MQT_NAMED_BUILDER(qir::multipleControlledRy)}));
