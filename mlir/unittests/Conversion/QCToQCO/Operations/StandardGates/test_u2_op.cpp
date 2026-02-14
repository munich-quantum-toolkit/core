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
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCU2OpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                                    MQT_NAMED_BUILDER(qco::u2)},
                    QCToQCOTestCase{"SingleControlledU2",
                                    MQT_NAMED_BUILDER(qc::singleControlledU2),
                                    MQT_NAMED_BUILDER(qco::singleControlledU2)},
                    QCToQCOTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qc::multipleControlledU2),
                        MQT_NAMED_BUILDER(qco::multipleControlledU2)}));
