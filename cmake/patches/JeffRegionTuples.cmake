# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The pinned jeff decoder assumes identical region input/output tuples. Remove this patch when the
# dependency decodes each tuple independently.
set(source_file "${SOURCE_DIR}/lib/Translation/Deserialize.cpp")
file(READ "${source_file}" contents)
foreach(kind IN ITEMS Switch While)
  if(kind STREQUAL "Switch")
    set(next "For")
  else()
    set(next "Scf")
  endif()
  string(FIND "${contents}" "void deserialize${kind}(" start)
  string(FIND "${contents}" "void deserialize${next}(" end)
  if(start LESS 0 OR end LESS start)
    message(FATAL_ERROR "Cannot locate the pinned jeff ${kind} decoder")
  endif()
  math(EXPR length "${end} - ${start}")
  string(SUBSTRING "${contents}" ${start} ${length} original)
  if(original MATCHES "inTypes")
    continue()
  endif()
  set(updated "${original}")
  string(REPLACE "llvm::SmallVector<mlir::Type> outTypes;"
                 "llvm::SmallVector<mlir::Type> inTypes, outTypes;" updated "${updated}")
  string(REPLACE "outTypes.reserve(" "inTypes.reserve(" updated "${updated}")
  string(REPLACE "outTypes.push_back(value.getType());" "inTypes.push_back(value.getType());"
                 updated "${updated}")
  string(
    REPLACE
      "    auto op = mlir::jeff::${kind}Op::create("
      "    for (const auto output : operation.getOutputs()) {
        outTypes.push_back(deserializeType(builder, ctx.getJeffType(output)));
    }

    auto op = mlir::jeff::${kind}Op::create("
      updated
      "${updated}")
  if(kind STREQUAL "While")
    string(REPLACE "op.getBefore().emplaceBlock(), outTypes"
                   "op.getBefore().emplaceBlock(), inTypes" updated "${updated}")
  else()
    string(REPLACE "emplaceBlock(), outTypes" "emplaceBlock(), inTypes" updated "${updated}")
  endif()
  string(REPLACE "${original}" "${updated}" contents "${contents}")
endforeach()
set(declaration
    "mlir::Type deserializeType(mlir::ImplicitLocOpBuilder& builder, const jeff::Type::Reader& type);

")
string(REPLACE "${declaration}" "" contents "${contents}")
string(REPLACE "void deserializeBlock(" "${declaration}void deserializeBlock(" contents
               "${contents}")
file(WRITE "${source_file}" "${contents}")
