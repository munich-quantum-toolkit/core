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
    QCOZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Z", MQT_NAMED_BUILDER(qco::z),
                                    MQT_NAMED_BUILDER(qc::z)},
                    QCOToQCTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledZ),
                                    MQT_NAMED_BUILDER(qc::singleControlledZ)},
                    QCOToQCTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledZ),
                        MQT_NAMED_BUILDER(qc::multipleControlledZ)}));
