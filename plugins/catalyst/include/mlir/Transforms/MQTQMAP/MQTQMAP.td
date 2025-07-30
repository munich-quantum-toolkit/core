// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTQMAP : Pass<"mqtqmap", "mlir::ModuleOp"> {
  let summary = "Map a quantum circuit to a device using MQT QMAP";
  let description = [{
    This pass uses the MQT QMAP library to map a quantum circuit onto a specified quantum device.
    The device topology is given via the `coupling-map` option as a string like "[[0,1],[1,0],[1,2]]".
  }];

  let options = [
Option<"couplingMap", "coupling-map", "std::string", "",
       "Coupling map as JSON-style string, e.g. [[0,1],[1,0],[1,2]]">
  ];
}
