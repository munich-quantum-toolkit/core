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
#include "qir_programs.h"
#include "test_qc_to_qir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCToQIRRZZOpTest, QCToQIRTest,
    testing::Values(
        QCToQIRTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                        MQT_NAMED_BUILDER(qir::rzz)},
        QCToQIRTestCase{"SingleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRzz),
                        MQT_NAMED_BUILDER(qir::singleControlledRzz)},
        QCToQIRTestCase{"MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qir::multipleControlledRzz)}));
