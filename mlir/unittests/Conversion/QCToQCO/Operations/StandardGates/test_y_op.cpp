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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCYOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                    MQT_NAMED_BUILDER(qco::y)},
                    QCToQCOTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qc::singleControlledY),
                                    MQT_NAMED_BUILDER(qco::singleControlledY)},
                    QCToQCOTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qc::multipleControlledY),
                        MQT_NAMED_BUILDER(qco::multipleControlledY)}));
