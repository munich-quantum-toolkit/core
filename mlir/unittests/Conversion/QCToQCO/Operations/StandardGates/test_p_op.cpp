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
    QCPOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                                    MQT_NAMED_BUILDER(qco::p)},
                    QCToQCOTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qc::singleControlledP),
                                    MQT_NAMED_BUILDER(qco::singleControlledP)},
                    QCToQCOTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qc::multipleControlledP),
                        MQT_NAMED_BUILDER(qco::multipleControlledP)}));
