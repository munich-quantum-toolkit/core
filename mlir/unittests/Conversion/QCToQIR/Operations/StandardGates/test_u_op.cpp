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
    QCToQIRUOpTest, QCToQIRTest,
    testing::Values(QCToQIRTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                    MQT_NAMED_BUILDER(qir::u)},
                    QCToQIRTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qc::singleControlledU),
                                    MQT_NAMED_BUILDER(qir::singleControlledU)},
                    QCToQIRTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qc::multipleControlledU),
                        MQT_NAMED_BUILDER(qir::multipleControlledU)}));
