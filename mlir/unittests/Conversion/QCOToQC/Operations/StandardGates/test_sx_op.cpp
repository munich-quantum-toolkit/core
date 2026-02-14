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
    QCOSXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"SX", MQT_NAMED_BUILDER(qco::sx),
                                    MQT_NAMED_BUILDER(qc::sx)},
                    QCOToQCTestCase{"SingleControlledSX",
                                    MQT_NAMED_BUILDER(qco::singleControlledSx),
                                    MQT_NAMED_BUILDER(qc::singleControlledSx)},
                    QCOToQCTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qco::multipleControlledSx),
                        MQT_NAMED_BUILDER(qc::multipleControlledSx)}));
