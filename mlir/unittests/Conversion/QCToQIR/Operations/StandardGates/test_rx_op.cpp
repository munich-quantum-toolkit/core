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
    QCToQIRRXOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                                    MQT_NAMED_BUILDER(qir::rx)},
                    QCToQIRTestCase{"SingleControlledRX",
                                    MQT_NAMED_BUILDER(qc::singleControlledRx),
                                    MQT_NAMED_BUILDER(qir::singleControlledRx)},
                    QCToQIRTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRx),
                        MQT_NAMED_BUILDER(qir::multipleControlledRx)}));
