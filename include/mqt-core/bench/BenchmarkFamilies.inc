/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file BenchmarkFamilies.inc
 * @brief Central registry of typed structured benchmark families.
 *
 * Define `MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)` before
 * including this file. `TYPE` is the C++ benchmark class. `STEM` names its
 * JSON and schema functions and its MLIR emitter. `ID` is the stable JSON
 * identifier. `DEFINITION_VERSION` identifies the benchmark semantics. Each
 * inclusion undefines `MQT_BENCHMARK_FAMILY`.
 *
 * Keep rows in benchmark ID order. The order is part of the benchmark list.
 */

#ifndef MQT_BENCHMARK_FAMILY
#define MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)
#endif

MQT_BENCHMARK_FAMILY(BV, bv, "bv", 1)
MQT_BENCHMARK_FAMILY(GHZ, ghz, "ghz", 1)
MQT_BENCHMARK_FAMILY(Grover, grover, "grover", 1)
MQT_BENCHMARK_FAMILY(QFT, qft, "qft", 1)
MQT_BENCHMARK_FAMILY(QPE, qpe, "qpe", 1)

#undef MQT_BENCHMARK_FAMILY
