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
    testing::Values(QCOToQCTestCase{"SingleMeasurementToSingleBit",
                                    qco::singleMeasurementToSingleBit,
                                    qc::singleMeasurementToSingleBit},
                    QCOToQCTestCase{"RepeatedMeasurementToSameBit",
                                    qco::repeatedMeasurementToSameBit,
                                    qc::repeatedMeasurementToSameBit},
                    QCOToQCTestCase{"RepeatedMeasurementToDifferentBits",
                                    qco::repeatedMeasurementToDifferentBits,
                                    qc::repeatedMeasurementToDifferentBits},
                    QCOToQCTestCase{
                        "MultipleClassicalRegistersAndMeasurements",
                        qco::multipleClassicalRegistersAndMeasurements,
                        qc::multipleClassicalRegistersAndMeasurements,
                    }),
    printTestName);
