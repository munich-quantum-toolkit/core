// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

//===- TensorBase.td - Base definitions for tensor dialect -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_BASE
#define TENSOR_BASE

include "mlir/IR/OpBase.td"

def QTensorDialect : Dialect {
  let name = "qtensor";
  let cppNamespace = "::mlir::qtensor";

  let description = [{
  TODO
  }];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "complex::ComplexDialect",
  ];
}

#endif // TENSOR_BASE
