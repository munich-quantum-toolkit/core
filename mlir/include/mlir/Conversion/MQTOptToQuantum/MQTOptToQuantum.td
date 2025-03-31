// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

//===-- Passes.td - Arith pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
include "mlir/Pass/PassBase.td"

def MQTOptToCatalystQuantum : Pass<"mqtopt-to-quantum"> {
  let summary = "Lower `mqtopt` to `quantum` dialect.";

  let description = [{
    This pass lowers the `mqtopt` to `quantum` dialect.
  }];
  let dependentDialects = [
    "catalyst::quantum::QuantumDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
