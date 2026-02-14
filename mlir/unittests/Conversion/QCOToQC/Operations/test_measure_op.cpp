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
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit)},
        QCOToQCTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit)},
        QCOToQCTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits)},
        QCOToQCTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements)}));
