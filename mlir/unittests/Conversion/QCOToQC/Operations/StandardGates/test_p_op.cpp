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
    QCOPOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"P", MQT_NAMED_BUILDER(qco::p),
                                    MQT_NAMED_BUILDER(qc::p)},
                    QCOToQCTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qco::singleControlledP),
                                    MQT_NAMED_BUILDER(qc::singleControlledP)},
                    QCOToQCTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qco::multipleControlledP),
                        MQT_NAMED_BUILDER(qc::multipleControlledP)}));
