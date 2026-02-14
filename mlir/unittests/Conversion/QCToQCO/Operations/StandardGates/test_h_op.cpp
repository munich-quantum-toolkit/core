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
    QCHOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                    MQT_NAMED_BUILDER(qco::h)},
                    QCToQCOTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qc::singleControlledH),
                                    MQT_NAMED_BUILDER(qco::singleControlledH)},
                    QCToQCOTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qc::multipleControlledH),
                        MQT_NAMED_BUILDER(qco::multipleControlledH)}));
