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
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCBarrierOpTest, QCTest,
    testing::Values(QCTestCase{"Barrier", MQT_NAMED_BUILDER(barrier),
                               MQT_NAMED_BUILDER(barrier)},
                    QCTestCase{"BarrierTwoQubits",
                               MQT_NAMED_BUILDER(barrierTwoQubits),
                               MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCTestCase{"BarrierMultipleQubits",
                               MQT_NAMED_BUILDER(barrierMultipleQubits),
                               MQT_NAMED_BUILDER(barrierMultipleQubits)},
                    QCTestCase{"SingleControlledBarrier",
                               MQT_NAMED_BUILDER(singleControlledBarrier),
                               MQT_NAMED_BUILDER(barrier)},
                    QCTestCase{"InverseBarrier",
                               MQT_NAMED_BUILDER(inverseBarrier),
                               MQT_NAMED_BUILDER(barrier)}));
