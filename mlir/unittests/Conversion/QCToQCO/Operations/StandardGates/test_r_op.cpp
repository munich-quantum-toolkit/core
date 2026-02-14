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
    QCROpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                    MQT_NAMED_BUILDER(qco::r)},
                    QCToQCOTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qc::singleControlledR),
                                    MQT_NAMED_BUILDER(qco::singleControlledR)},
                    QCToQCOTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qc::multipleControlledR),
                        MQT_NAMED_BUILDER(qco::multipleControlledR)}));
