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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCMeasureOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit)},
        QCToQCOTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit)},
        QCToQCOTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits)},
        QCToQCOTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(
                qco::multipleClassicalRegistersAndMeasurements)}));
