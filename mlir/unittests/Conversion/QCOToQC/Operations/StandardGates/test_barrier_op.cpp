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
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Barrier", MQT_NAMED_BUILDER(qco::barrier),
                                    MQT_NAMED_BUILDER(qc::barrier)},
                    QCOToQCTestCase{"BarrierTwoQubits",
                                    MQT_NAMED_BUILDER(qco::barrierTwoQubits),
                                    MQT_NAMED_BUILDER(qc::barrierTwoQubits)},
                    QCOToQCTestCase{
                        "BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qco::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits)}));
