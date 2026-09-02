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

def MQTDialect : Dialect {
  let name = "mqt";
  let cppNamespace = "::mlir::mqt";
  let useDefaultAttributePrinterParser = 1;

  let summary = "Shared metadata for MQT quantum programs.";
  let description = [{
    The MQT dialect owns frontend-neutral metadata that must remain meaningful
    across quantum dialect conversions. It defines no operations or types.

    `mqt.input_name` records the source-level name of a function input.
    `mqt.parameter_group` optionally preserves the source-level vector
    identity, name, element index, and size of a function input or a lexically
    bound `scf.for` parameter.
    `mqt.register_name` records the source-level name of a quantum or classical
    register allocation. Input and register names share one function-wide
    namespace.
    `mqt.entry_point` marks the single defined program entry function in a
    module.
    `mqt.unitary` marks a private function that defines a unitary operation.
    `#mqt.compilation_target` records compiler-target facts as typed IR.
  }];

  let discardableAttrs = (ins "::mlir::StringAttr":$input_name,
      "::mlir::DictionaryAttr":$parameter_group,
      "::mlir::StringAttr":$register_name, "::mlir::UnitAttr":$entry_point,
      "::mlir::UnitAttr":$unitary);

  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
}

class MQTAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MQTDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class MQTArrayRefParameter<string element, string desc = "">
    : ArrayRefParameter<element, desc> {
  let parser = "::mlir::mqt::detail::parseArray<"#element#">($_parser)";
  let printer = "::mlir::mqt::detail::printArray($_printer, $_self)";
}

class MQTOptionalUInt64Parameter
    : OptionalParameter<"std::optional<uint64_t>"> {
  let parser = "::mlir::FieldParser<uint64_t>::parse($_parser)";
}

def ConnectivityKind
    : I32EnumAttr<"ConnectivityKind", "Target connectivity",
                  [I32EnumAttrCase<"AllToAll", 0, "all_to_all">,
                   I32EnumAttrCase<"Explicit", 1, "explicit">]> {
  let cppNamespace = "::mlir::mqt";
  let genSpecializedAttr = 0;
}

def NativeOperationsKind
    : I32EnumAttr<"NativeOperationsKind", "Native-operation support",
                  [I32EnumAttrCase<"Unrestricted", 0, "unrestricted">,
                   I32EnumAttrCase<"Explicit", 1, "explicit">]> {
  let cppNamespace = "::mlir::mqt";
  let genSpecializedAttr = 0;
}

def OperationArityKind
    : I32EnumAttr<"OperationArityKind", "Operation arity kind",
                  [I32EnumAttrCase<"Fixed", 0, "fixed">,
                   I32EnumAttrCase<"Variadic", 1, "variadic">]> {
  let cppNamespace = "::mlir::mqt";
  let genSpecializedAttr = 0;
}

def DurationUnitAttr : MQTAttr<"DurationUnit", "duration_unit"> {
  let summary = "Unit for raw compiler-target durations";
  let description = [{
    The scale factor converts each raw duration to the named unit. For example,
    `#mqt.duration_unit<unit = "ns", scale_factor = 1.000000e+00 : f64>`
    records durations in nanoseconds.
  }];
  let parameters = (ins "StringAttr":$unit, "FloatAttr":$scale_factor);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def SiteAttr : MQTAttr<"Site", "site"> {
  let summary = "One ordered compiler-target site";
  let description = [{
    Site identifiers come from the target. The array order defines dense
    compiler vertices. For example, `#mqt.site<id = 4, name = "q4">` records
    one named site.
  }];
  let parameters = (ins "int64_t":$id, OptionalParameter<"StringAttr">:$name,
      MQTOptionalUInt64Parameter<>:$t1, MQTOptionalUInt64Parameter<>:$t2);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def CouplingAttr : MQTAttr<"Coupling", "coupling"> {
  let summary = "One undirected compiler-target coupling";
  let description = [{
    A coupling joins two target site identifiers. For example,
    `#mqt.coupling<source = 4, target = 7>` permits interactions between sites
    4 and 7.
  }];
  let parameters = (ins "int64_t":$source, "int64_t":$target);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def SiteTupleAttr : MQTAttr<"SiteTuple", "site_tuple"> {
  let summary = "Calibration data for an ordered site tuple";
  let description = [{
    The tuple defines one placement of a native operation. For example,
    `#mqt.site_tuple<sites = [4, 7], duration = 40>` records an ordered
    two-site placement with a raw duration of 40.
  }];
  let parameters = (ins MQTArrayRefParameter<"int64_t">:$sites,
      MQTOptionalUInt64Parameter<>:$duration,
      OptionalParameter<"FloatAttr">:$fidelity);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def OperationArityAttr : MQTAttr<"OperationArity", "operation_arity"> {
  let summary = "Accepted width of a compiler-target operation";
  let description = [{
    A fixed arity accepts exactly its value. A variadic arity accepts its value
    or any larger width. For example,
    `#mqt.operation_arity<kind = variadic, value = 1>` accepts a gate with one
    target and any number of controls.
  }];
  let parameters = (ins EnumParameter<OperationArityKind>:$kind,
      "uint64_t":$value);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def NativeOperationAttr : MQTAttr<"NativeOperation", "native_operation"> {
  let summary = "One native compiler-target operation";
  let description = [{
    The operation records its spelling, arity, parameter count, and optional
    global or site-specific calibration data. The following example records a
    placed controlled-X operation:

    ```mlir
    #mqt.native_operation<name = "cx",
        arity = #mqt.operation_arity<kind = fixed, value = 2>,
        num_parameters = 0, site_tuples = [<sites = [4, 7]>]>
    ```
  }];
  let parameters = (ins "StringAttr":$name, "OperationArityAttr":$arity,
      "uint64_t":$num_parameters,
      MQTArrayRefParameter<"SiteTupleAttr">:$site_tuples,
      MQTOptionalUInt64Parameter<>:$duration,
      OptionalParameter<"FloatAttr">:$fidelity);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def CompilationTargetAttr : MQTAttr<"CompilationTarget", "compilation_target"> {
  let summary = "Typed compiler-target facts";
  let description = [{
    Records source target facts without derived routing or synthesis caches.
    Ordered sites define dense compiler vertices. Connectivity and native
    operation support is unrestricted or explicit.

    The following target has two explicitly connected sites and one native
    operation:

    ```mlir
    #mqt.compilation_target<
        name = "device", sites = [<id = 4>, <id = 7>],
        connectivity = explicit, couplings = [<source = 4, target = 7>],
        native_operations = explicit,
        operations = [<name = "cx",
            arity = #mqt.operation_arity<kind = fixed, value = 2>,
            num_parameters = 0, site_tuples = []>]>
    ```
  }];
  let parameters = (ins OptionalParameter<"StringAttr">:$name,
      MQTArrayRefParameter<"SiteAttr">:$sites,
      OptionalParameter<"DurationUnitAttr">:$duration_unit,
      EnumParameter<ConnectivityKind>:$connectivity,
      MQTArrayRefParameter<"CouplingAttr">:$couplings,
      EnumParameter<NativeOperationsKind>:$native_operations,
      MQTArrayRefParameter<"NativeOperationAttr">:$operations);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
