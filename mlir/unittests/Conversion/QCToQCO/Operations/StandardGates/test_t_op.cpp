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
    QCTOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                                    MQT_NAMED_BUILDER(qco::t_)},
                    QCToQCOTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qc::singleControlledT),
                                    MQT_NAMED_BUILDER(qco::singleControlledT)},
                    QCToQCOTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qc::multipleControlledT),
                        MQT_NAMED_BUILDER(qco::multipleControlledT)}));
