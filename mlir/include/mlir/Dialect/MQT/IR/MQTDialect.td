// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
#define MLIR_DIALECT_MQT_IR_MQTDIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
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
    `mqt.target_env` records compiler-target facts and the selected payload
    execution contract.
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

class MQT_ArrayRefParameter<string element, string desc = "">
    : ArrayRefParameter<element, desc> {
  let parser = "::mlir::mqt::detail::parseArray<"#element#">($_parser)";
  let printer = "::mlir::mqt::detail::printArray($_printer, $_self)";
}

def MQT_PayloadEncoding
    : I32EnumAttr<"PayloadEncoding", "Payload encoding",
                  [I32EnumAttrCase<"Text", 0, "text">,
                   I32EnumAttrCase<"Binary", 1, "binary">]> {
  let cppNamespace = "::mlir::mqt";
  let genSpecializedAttr = 0;
}

def MQT_ConnectivityKind
    : I32EnumAttr<"ConnectivityKind", "Connectivity knowledge",
                  [I32EnumAttrCase<"Unknown", 0, "unknown">,
                   I32EnumAttrCase<"AllToAll", 1, "all_to_all">,
                   I32EnumAttrCase<"Explicit", 2, "explicit">]> {
  let cppNamespace = "::mlir::mqt";
  let genSpecializedAttr = 0;
}

def MQT_NativeOperationsKind
    : I32EnumAttr<"NativeOperationsKind", "Native-operation knowledge",
                  [I32EnumAttrCase<"Unknown", 0, "unknown">,
                   I32EnumAttrCase<"Unrestricted", 1, "unrestricted">,
                   I32EnumAttrCase<"Explicit", 2, "explicit">]> {
  let cppNamespace = "::mlir::mqt";
  let genSpecializedAttr = 0;
}

def MQT_PayloadDescriptorAttr
    : MQT_Attr<"PayloadDescriptor", "payload_descriptor"> {
  let summary = "Exact payload identity";
  let description = [{
    Identifies a payload by format ID, exact version, optional profile ID, and
    encoding. Every field takes part in identity. An empty profile denotes a
    payload without a named profile.
  }];
  let parameters = (ins "StringAttr":$id, "StringAttr":$version,
      "StringAttr":$profile, EnumParameter<MQT_PayloadEncoding>:$encoding);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_ProgramConstraintAttr
    : MQT_Attr<"ProgramConstraint", "program_constraint"> {
  let summary = "One constraint on a payload capability";
  let description = [{
    Records one extensible constraint ID and its constraint-specific value.
  }];
  let parameters = (ins "StringAttr":$id, "uint64_t":$value);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_ProgramCapabilityAttr
    : MQT_Attr<"ProgramCapability", "program_capability"> {
  let summary = "One payload execution capability";
  let description = [{
    Records one extensible capability ID, its feature-specific value, and all
    constraints on that capability. Unknown IDs remain valid so newer
    producers can round-trip through older consumers.
  }];
  let parameters = (ins "StringAttr":$id, "uint64_t":$value,
      MQT_ArrayRefParameter<"ProgramConstraintAttr">:$constraints);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_PayloadEnvAttr : MQT_Attr<"PayloadEnv", "payload_env"> {
  let summary = "Selected payload execution contract";
  let description = [{
    Records the exact payload and its effective capabilities. Producers expand
    descriptor baselines before creating this attribute. The capability list
    remains available when optional capability metadata is unknown;
    `optional_capabilities_known` records whether that optional metadata is
    complete.
  }];
  let parameters = (ins "PayloadDescriptorAttr":$descriptor,
      MQT_ArrayRefParameter<"ProgramCapabilityAttr">:$capabilities,
      "bool":$optional_capabilities_known);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_DurationUnitAttr : MQT_Attr<"DurationUnit", "duration_unit"> {
  let summary = "Unit for raw compiler-target durations";
  let parameters = (ins "StringAttr":$unit, "FloatAttr":$scale_factor);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_SiteAttr : MQT_Attr<"Site", "site"> {
  let summary = "One ordered compiler-target site";
  let parameters = (ins "int64_t":$id, OptionalParameter<"StringAttr">:$name,
      OptionalParameter<"std::optional<uint64_t>">:$t1,
      OptionalParameter<"std::optional<uint64_t>">:$t2);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_CouplingAttr : MQT_Attr<"Coupling", "coupling"> {
  let summary = "One undirected compiler-target coupling";
  let parameters = (ins "int64_t":$source, "int64_t":$target);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_SiteTupleAttr : MQT_Attr<"SiteTuple", "site_tuple"> {
  let summary = "Calibration data for an ordered site tuple";
  let parameters = (ins MQT_ArrayRefParameter<"int64_t">:$sites,
      OptionalParameter<"std::optional<uint64_t>">:$duration,
      OptionalParameter<"FloatAttr">:$fidelity);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_NativeOperationAttr : MQT_Attr<"NativeOperation", "native_operation"> {
  let summary = "One native compiler-target operation";
  let parameters = (ins "StringAttr":$name, "uint64_t":$arity,
      "uint64_t":$num_parameters,
      MQT_ArrayRefParameter<"SiteTupleAttr">:$site_tuples,
      OptionalParameter<"std::optional<uint64_t>">:$duration,
      OptionalParameter<"FloatAttr">:$fidelity);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_CompilationTargetAttr
    : MQT_Attr<"CompilationTarget", "compilation_target"> {
  let summary = "Typed compiler-target facts";
  let description = [{
    Records source target facts without derived routing or synthesis caches.
    Ordered sites define dense compiler vertices. Connectivity and native
    operations distinguish unknown, unrestricted, and explicit knowledge.
  }];
  let parameters = (ins OptionalParameter<"StringAttr">:$name,
      MQT_ArrayRefParameter<"SiteAttr">:$sites,
      OptionalParameter<"DurationUnitAttr">:$duration_unit,
      EnumParameter<MQT_ConnectivityKind>:$connectivity,
      MQT_ArrayRefParameter<"CouplingAttr">:$couplings,
      EnumParameter<MQT_NativeOperationsKind>:$native_operations,
      MQT_ArrayRefParameter<"NativeOperationAttr">:$operations);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def MQT_TargetEnvAttr
    : MQT_Attr<"TargetEnv", "target_env", [DLTIQueryInterface]> {
  let summary = "Selected compilation and payload environment";
  let description = [{
    Combines typed compiler-target facts with the selected payload contract.
    Optional DLTI extensions carry provider-specific facts. Direct access to
    the typed fields is authoritative.
  }];
  let parameters = (ins "CompilationTargetAttr":$compilation_target,
      "PayloadEnvAttr":$payload_env,
      OptionalParameter<"::mlir::MapAttr">:$extensions);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral kCompilationTargetKey =
        "mqt.compilation_target";
    static constexpr ::llvm::StringLiteral kPayloadEnvKey = "mqt.payload_env";

    ::mlir::FailureOr<::mlir::Attribute>
    query(::mlir::DataLayoutEntryKey key) const;
  }];
}

#endif // MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
