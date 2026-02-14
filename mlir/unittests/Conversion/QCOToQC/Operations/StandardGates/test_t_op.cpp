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
    QCOTOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"T", MQT_NAMED_BUILDER(qco::t_),
                                    MQT_NAMED_BUILDER(qc::t_)},
                    QCOToQCTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qco::singleControlledT),
                                    MQT_NAMED_BUILDER(qc::singleControlledT)},
                    QCOToQCTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qco::multipleControlledT),
                        MQT_NAMED_BUILDER(qc::multipleControlledT)}));
