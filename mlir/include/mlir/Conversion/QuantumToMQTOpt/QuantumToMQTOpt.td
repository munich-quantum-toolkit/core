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

#ifndef LIB_CONVERSION_QUANTUMTOMQTOPT
#define LIB_CONVERSION_QUANTUMTOMQTOPT

include "mlir/Pass/PassBase.td"

def CatalystQuantumToMQTOpt : Pass<"quantum-to-mqtopt"> {
  let summary = "Lower `quantum` to `mqtopt` dialect.";

  let description = [{
    This pass lowers the `quantum` to `mqtopt` dialect.
  }];


  // Define dependent dialects
  let dependentDialects = [
    "catalyst::quantum::QuantumDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}

#endif  // LIB_CONVERSION_QUANTUMTOMQTOPT
