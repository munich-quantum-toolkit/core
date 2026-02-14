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
    QCXXPlusYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                        MQT_NAMED_BUILDER(qco::xxPlusYY)},
        QCToQCOTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY)},
        QCToQCOTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY)}));
