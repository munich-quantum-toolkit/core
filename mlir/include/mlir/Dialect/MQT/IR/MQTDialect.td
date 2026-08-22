// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
#define MLIR_DIALECT_MQT_IR_MQTDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

def MQTDialect : Dialect {
  let name = "mqt";
  let cppNamespace = "::mlir::mqt";
  let useDefaultAttributePrinterParser = 1;

  let summary = "Shared metadata for MQT quantum programs.";
  let description = [{
    The MQT dialect owns frontend-neutral metadata that must remain meaningful
    across quantum dialect conversions. It defines no operations or types.

    `mqt.input_name` records the source-level name of a function input.
    `mqt.register_name` records the source-level name of a quantum or classical
    register allocation. Input and register names share one function-wide
    namespace.
    `mqt.entry_point` marks the single defined program entry function in a
    module.
    `mqt.target_env` records the selected compiler payload and its execution
    profile as reproducible module metadata.
  }];

  let discardableAttrs = (ins "::mlir::StringAttr":$input_name,
      "::mlir::StringAttr":$register_name, "::mlir::UnitAttr":$entry_point);

  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
}

class MQT_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MQTDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def MQT_TargetEnvAttr
    : MQT_Attr<"TargetEnv", "target_env", [DLTIQueryInterface]> {
  let summary = "Selected compiler payload execution profile";
  let description = [{
    Records the selected compiler payload, the target features known to be
    supported for that payload, and whether the target reported all optional
    features. This is execution-environment metadata, not a data-layout
    specification.
  }];

  let parameters = (ins "StringAttr":$id, "StringAttr":$version,
      "StringAttr":$profile, "StringAttr":$encoding, "ArrayAttr":$capabilities,
      "bool":$optionalFeaturesKnown, "DictionaryAttr":$target);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral getOperationAttributeName() {
      return "mqt.target_env";
    }

    static TargetEnvAttr get(::mlir::MLIRContext* context,
                             ::llvm::StringRef id,
                             ::llvm::StringRef version,
                             ::llvm::StringRef profile,
                             ::llvm::StringRef encoding,
                             ::llvm::ArrayRef<std::pair<::llvm::StringRef,
                                                        uint64_t>> capabilities,
                             bool optionalFeaturesKnown,
                             ::mlir::DictionaryAttr target);

    [[nodiscard]] bool supports(::llvm::StringRef feature,
                                uint64_t value = 0U) const;

    ::mlir::FailureOr<::mlir::Attribute>
    query(::mlir::DataLayoutEntryKey key);
  }];
}

#endif // MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
