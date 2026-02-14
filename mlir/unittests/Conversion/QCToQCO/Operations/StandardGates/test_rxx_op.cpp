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
    QCRXXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                        MQT_NAMED_BUILDER(qco::rxx)},
        QCToQCOTestCase{"SingleControlledRXX",
                        MQT_NAMED_BUILDER(qc::singleControlledRxx),
                        MQT_NAMED_BUILDER(qco::singleControlledRxx)},
        QCToQCOTestCase{"MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRxx)}));
