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
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, QCOTest,
    testing::Values(QCOTestCase{"Barrier", MQT_NAMED_BUILDER(barrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"BarrierTwoQubits",
                                MQT_NAMED_BUILDER(barrierTwoQubits),
                                MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCOTestCase{"BarrierMultipleQubits",
                                MQT_NAMED_BUILDER(barrierMultipleQubits),
                                MQT_NAMED_BUILDER(barrierMultipleQubits)},
                    QCOTestCase{"SingleControlledBarrier",
                                MQT_NAMED_BUILDER(singleControlledBarrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"InverseBarrier",
                                MQT_NAMED_BUILDER(inverseBarrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"TwoBarrier", MQT_NAMED_BUILDER(twoBarrier),
                                MQT_NAMED_BUILDER(barrierTwoQubits)}));
