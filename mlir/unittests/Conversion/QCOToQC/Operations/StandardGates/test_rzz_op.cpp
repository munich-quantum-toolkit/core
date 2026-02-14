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
    QCORZZOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"RZZ", MQT_NAMED_BUILDER(qco::rzz),
                                    MQT_NAMED_BUILDER(qc::rzz)},
                    QCOToQCTestCase{"SingleControlledRZZ",
                                    MQT_NAMED_BUILDER(qco::singleControlledRzz),
                                    MQT_NAMED_BUILDER(qc::singleControlledRzz)},
                    QCOToQCTestCase{
                        "MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qco::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz)}));
