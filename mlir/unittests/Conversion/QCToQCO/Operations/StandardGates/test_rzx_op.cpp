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
    QCRZXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                        MQT_NAMED_BUILDER(qco::rzx)},
        QCToQCOTestCase{"SingleControlledRZX",
                        MQT_NAMED_BUILDER(qc::singleControlledRzx),
                        MQT_NAMED_BUILDER(qco::singleControlledRzx)},
        QCToQCOTestCase{"MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRzx)}));
