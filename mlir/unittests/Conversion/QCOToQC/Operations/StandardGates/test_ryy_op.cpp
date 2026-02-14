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
    QCORYYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RYY", MQT_NAMED_BUILDER(qco::ryy),
                                    MQT_NAMED_BUILDER(qc::ryy)},
                    QCOToQCTestCase{"SingleControlledRYY",
                                    MQT_NAMED_BUILDER(qco::singleControlledRyy),
                                    MQT_NAMED_BUILDER(qc::singleControlledRyy)},
                    QCOToQCTestCase{
                        "MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy)}));
