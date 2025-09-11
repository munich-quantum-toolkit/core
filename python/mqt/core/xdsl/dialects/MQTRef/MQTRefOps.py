# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from xdsl.dialects.builtin import f64, i1
from xdsl.ir import Dialect, TypeAttribute
from xdsl.irdl import (
    Attribute,
    Data,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    var_operand_def,
)


# Custom Qubit type for MQTRef dialect (reference semantics)
@irdl_attr_definition
class QubitType(Data[None], TypeAttribute):
    name = "mqtref.Qubit"

    @classmethod
    def parse_parameter(cls, parser) -> None:
        return None

    def print_parameter(self, printer) -> None:
        pass


# Custom QubitRegister type for MQTRef dialect (reference semantics)
@irdl_attr_definition
class QregType(Data[None], TypeAttribute):
    name = "mqtref.QubitRegister"

    @classmethod
    def parse_parameter(cls, parser) -> None:
        return None

    def print_parameter(self, printer) -> None:
        pass


# Base class for all MQTRef operations
class MQTRefOp(IRDLOperation):
    """Base class for all operations in the MQTRef dialect."""


# Base class for gate operations (equivalent to GateOp in TableGen)
class GateOp(MQTRefOp):
    """Base class for gate operations in reference semantics."""


# Base class for unitary operations (equivalent to UnitaryOp in TableGen)
class UnitaryOp(GateOp):
    """Base class for unitary operations with reference semantics."""

    # Note: In TableGen this has AttrSizedOperandSegments, UnitaryInterface traits
    # Reference semantics means no results - operations modify qubits in place

    static_params = attr_def(Attribute, attr_name="static_params")
    params_mask = attr_def(Attribute, attr_name="params_mask")
    params = var_operand_def(f64)
    in_qubits = var_operand_def(QubitType)
    pos_ctrl_in_qubits = var_operand_def(QubitType)
    neg_ctrl_in_qubits = var_operand_def(QubitType)

    # No output qubits in reference semantics - operations modify qubits in place


# Base class for resource operations
class ResourceOp(MQTRefOp):
    """Base class for resource operations."""


# MeasureOp implementation as a GateOp (reference semantics)
@irdl_op_definition
class MeasureOp(GateOp):
    name = "mqtref.measure"
    traits = traits_def()  # No special traits

    in_qubit = operand_def(QubitType)
    out_bit = result_def(i1)


# XOp implementation as a simple GateOp (reference semantics - minimum working example)
@irdl_op_definition
class XOp(GateOp):
    name = "mqtref.x"
    traits = traits_def()  # OneTarget, NoParameter traits would be here

    # Reference semantics: operates on qubit in place, no output qubit
    qubit = operand_def(QubitType)


# AllocQubitOp implementation as a ResourceOp
@irdl_op_definition
class AllocQubitOp(ResourceOp):
    name = "mqtref.allocQubit"
    traits = traits_def()

    qubit = result_def(QubitType)


# DeallocQubitOp implementation as a ResourceOp
@irdl_op_definition
class DeallocQubitOp(ResourceOp):
    name = "mqtref.deallocQubit"
    traits = traits_def()

    qubit = operand_def(QubitType)


# MQTRef dialect definition
MQTRef = Dialect(
    "mqtref",
    [
        MeasureOp,
        XOp,
        AllocQubitOp,
        DeallocQubitOp,
    ],
    [
        QubitType,
        QregType,
    ],
)
