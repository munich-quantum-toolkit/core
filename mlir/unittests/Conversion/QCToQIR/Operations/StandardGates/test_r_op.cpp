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
    QCToQIRROpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                    MQT_NAMED_BUILDER(qir::r)},
                    QCToQIRTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qc::singleControlledR),
                                    MQT_NAMED_BUILDER(qir::singleControlledR)},
                    QCToQIRTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qc::multipleControlledR),
                        MQT_NAMED_BUILDER(qir::multipleControlledR)}));
