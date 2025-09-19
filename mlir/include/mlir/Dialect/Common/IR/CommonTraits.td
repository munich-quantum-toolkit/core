// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTCOMMON_TRAITS
#define MQTCOMMON_TRAITS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class DecimalFraction<int BeforeSeparator, int AfterSeparator> {
  int Before = BeforeSeparator;
  int After = AfterSeparator;
}
class Int<int I> : DecimalFraction<I, 0>;

class DefinitionElementType<int Id> {
  int id = Id;
}
def Value : DefinitionElementType<0>;
def ParameterIndex : DefinitionElementType<1>;

class FunctionType<int Id> {
  int id = Id;
}

def Identity : FunctionType<0>;
def Sin : FunctionType<1>;
def Cos : FunctionType<2>;

class DefinitionElement<DecimalFraction V, DefinitionElementType T, FunctionType F = Identity> {
  DecimalFraction v = V;
  DefinitionElementType t = T;
  FunctionType f = F;
}

class Value<DecimalFraction V> : DefinitionElement<V, Value, Identity>;

class TargetArity<int N, list<DefinitionElement> MatrixDefinition = []>
    : ParamNativeOpTrait<
        "TargetArityTrait",
        !cast<string>(N) # ", {" # !foldl("", MatrixDefinition, acc, var, "{ " # var.v.Before # "." # var.v.After # ", " # !cond(!eq(var.t, Value) : "mqt::ir::common::DefinitionMatrixElement::Type::Value") # ", " # !cond(!eq(var.f, Identity) : "mqt::ir::common::DefinitionMatrixElement::Transformation::Identity") # " }") # "}"
    > {
    let cppNamespace = "::mqt::ir::common";
}

def NoTarget : TargetArity<0>;
def OneTarget : TargetArity<1>;
def TwoTarget : TargetArity<2>;

class ParameterArity<int N>
    : ParamNativeOpTrait<"ParameterArityTrait", !cast<string>(N)> {
    let cppNamespace = "::mqt::ir::common";
}

def NoParameter : ParameterArity<0>;
def OneParameter : ParameterArity<1>;
def TwoParameters : ParameterArity<2>;
def ThreeParameters : ParameterArity<3>;

def NoControl : NativeOpTrait<"NoControlTrait"> {
    let cppNamespace = "::mqt::ir::common";
}

def UniqueSizeDefinition : NativeOpTrait<"UniqueSizeDefinitionTrait"> {
    let cppNamespace = "::mqt::ir::common";
}

def UniqueIndexDefinition : NativeOpTrait<"UniqueIndexDefinitionTrait"> {
    let cppNamespace = "::mqt::ir::common";
}

#endif // MQTCOMMON_TRAITS
