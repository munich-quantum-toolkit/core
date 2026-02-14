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
    QCZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                                    MQT_NAMED_BUILDER(qco::z)},
                    QCToQCOTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledZ),
                                    MQT_NAMED_BUILDER(qco::singleControlledZ)},
                    QCToQCOTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledZ),
                        MQT_NAMED_BUILDER(qco::multipleControlledZ)}));
