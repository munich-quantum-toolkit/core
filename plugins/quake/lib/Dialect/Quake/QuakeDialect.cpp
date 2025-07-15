/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mlir/Dialect/Quake/QuakeDialect.h"

#include "mlir/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Quake/QuakeTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quake/QuakeDialect.cpp.inc"

//===----------------------------------------------------------------------===//

void quake::QuakeDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quake/QuakeOps.cpp.inc"
      >();
  addInterfaces<QuakeInlinerInterface>();
}
