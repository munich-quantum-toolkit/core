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
    QCOXOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"X", MQT_NAMED_BUILDER(qco::x),
                                    MQT_NAMED_BUILDER(qc::x)},
                    QCOToQCTestCase{"SingleControlledX",
                                    MQT_NAMED_BUILDER(qco::singleControlledX),
                                    MQT_NAMED_BUILDER(qc::singleControlledX)},
                    QCOToQCTestCase{
                        "MultipleControlledX",
                        MQT_NAMED_BUILDER(qco::multipleControlledX),
                        MQT_NAMED_BUILDER(qc::multipleControlledX)}));
