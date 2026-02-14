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
    QCORYOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RY", MQT_NAMED_BUILDER(qco::ry),
                                    MQT_NAMED_BUILDER(qc::ry)},
                    QCOToQCTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qco::singleControlledRy),
                                    MQT_NAMED_BUILDER(qc::singleControlledRy)},
                    QCOToQCTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qco::multipleControlledRy),
                        MQT_NAMED_BUILDER(qc::multipleControlledRy)}));
