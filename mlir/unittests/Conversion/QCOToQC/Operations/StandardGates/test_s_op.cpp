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
    QCOSOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"S", MQT_NAMED_BUILDER(qco::s),
                                    MQT_NAMED_BUILDER(qc::s)},
                    QCOToQCTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qco::singleControlledS),
                                    MQT_NAMED_BUILDER(qc::singleControlledS)},
                    QCOToQCTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qco::multipleControlledS),
                        MQT_NAMED_BUILDER(qc::multipleControlledS)}));
