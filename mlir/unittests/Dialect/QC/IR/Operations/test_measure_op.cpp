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
    QCMeasureOpTest, QCTest,
    testing::Values(
        QCTestCase{"SingleMeasurementToSingleBit",
                   MQT_NAMED_BUILDER(singleMeasurementToSingleBit),
                   MQT_NAMED_BUILDER(singleMeasurementToSingleBit)},
        QCTestCase{"RepeatedMeasurementToSameBit",
                   MQT_NAMED_BUILDER(repeatedMeasurementToSameBit),
                   MQT_NAMED_BUILDER(repeatedMeasurementToSameBit)},
        QCTestCase{"RepeatedMeasurementToDifferentBits",
                   MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits),
                   MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits)},
        QCTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements)}));
