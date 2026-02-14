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
    QCToQIRYOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                    MQT_NAMED_BUILDER(qir::y)},
                    QCToQIRTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qc::singleControlledY),
                                    MQT_NAMED_BUILDER(qir::singleControlledY)},
                    QCToQIRTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qc::multipleControlledY),
                        MQT_NAMED_BUILDER(qir::multipleControlledY)}));
