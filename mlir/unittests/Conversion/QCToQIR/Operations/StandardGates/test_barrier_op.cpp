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
using namespace mlir::qir;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRBarrierOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"Barrier", MQT_NAMED_BUILDER(qc::barrier),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"BarrierTwoQubits",
                        MQT_NAMED_BUILDER(qc::barrierTwoQubits),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(emptyQIR)},
        QCToQIRTestCase{"SingleControlledBarrier",
                        MQT_NAMED_BUILDER(qc::singleControlledBarrier),
                        MQT_NAMED_BUILDER(emptyQIR)}));
