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
    QCOYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Y", MQT_NAMED_BUILDER(qco::y),
                                    MQT_NAMED_BUILDER(qc::y)},
                    QCOToQCTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qco::singleControlledY),
                                    MQT_NAMED_BUILDER(qc::singleControlledY)},
                    QCOToQCTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qco::multipleControlledY),
                        MQT_NAMED_BUILDER(qc::multipleControlledY)}));
