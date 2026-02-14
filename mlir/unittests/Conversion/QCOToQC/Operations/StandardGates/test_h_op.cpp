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
    QCOHOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"H", MQT_NAMED_BUILDER(qco::h),
                                    MQT_NAMED_BUILDER(qc::h)},
                    QCOToQCTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qco::singleControlledH),
                                    MQT_NAMED_BUILDER(qc::singleControlledH)},
                    QCOToQCTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qco::multipleControlledH),
                        MQT_NAMED_BUILDER(qc::multipleControlledH)}));
