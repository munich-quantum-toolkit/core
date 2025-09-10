from xdsl.dialects.builtin import i1, f64, i64
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    result_def,
    var_operand_def,
    var_result_def,
    attr_def,
    irdl_attr_definition,
    Data,
    Attribute,
    traits_def,
)
from xdsl.traits import NoMemoryEffect


# Custom Qubit type for MQTOpt dialect
@irdl_attr_definition
class QubitType(Data):
    name = "mqtopt.Qubit"


# Custom QubitRegister type for MQTOpt dialect
@irdl_attr_definition
class QregType(Data):
    name = "mqtopt.QubitRegister"


# Base class for all MQTOpt operations
class MQTOptOp(IRDLOperation):
    """Base class for all operations in the MQTOpt dialect."""
    pass


# Base class for gate operations (equivalent to GateOp in TableGen)
class GateOp(MQTOptOp):
    """Base class for gate operations with NoMemoryEffect trait."""
    traits = traits_def(NoMemoryEffect())


# Base class for unitary operations (equivalent to UnitaryOp in TableGen)
class UnitaryOp(GateOp):
    """Base class for unitary operations with variadic operands and results."""
    # Note: In TableGen this has AttrSizedOperandSegments, AttrSizedResultSegments, UnitaryInterface traits
    
    static_params = attr_def(Attribute, attr_name="static_params")
    params_mask = attr_def(Attribute, attr_name="params_mask") 
    params = var_operand_def(f64)
    in_qubits = var_operand_def(QubitType)
    pos_ctrl_in_qubits = var_operand_def(QubitType)
    neg_ctrl_in_qubits = var_operand_def(QubitType)
    
    out_qubits = var_result_def(QubitType)
    pos_ctrl_out_qubits = var_result_def(QubitType)
    neg_ctrl_out_qubits = var_result_def(QubitType)


# Base class for resource operations
class ResourceOp(MQTOptOp):
    """Base class for resource operations."""
    pass


# MeasureOp implementation as a GateOp
@irdl_op_definition
class MeasureOp(GateOp):
    name = "mqtopt.measure"
    traits = traits_def()  # Empty traits list as in TableGen MeasureOp : GateOp<"measure", []>
    
    in_qubit = operand_def(QubitType)
    out_qubit = result_def(QubitType)
    out_bit = result_def(i1)


# XOp implementation as a simple GateOp 
@irdl_op_definition
class XOp(GateOp):
    name = "mqtopt.x"
    traits = traits_def()  # OneTarget, NoParameter traits would be here
    
    in_qubit = operand_def(QubitType)
    out_qubit = result_def(QubitType)
    

# AllocOp implementation as a ResourceOp (allocQubitRegister)
@irdl_op_definition
class AllocOp(ResourceOp):
    name = "mqtopt.allocQubitRegister"
    traits = traits_def()  # UniqueSizeDefinition trait would be here
    
    size = opt_operand_def(i64)
    size_attr = attr_def(Attribute, attr_name="size_attr")
    qreg = result_def(QregType)


# DeallocOp implementation as a ResourceOp (deallocQubitRegister)
@irdl_op_definition
class DeallocOp(ResourceOp):
    name = "mqtopt.deallocQubitRegister"
    traits = traits_def()
    
    qreg = operand_def(QregType)


# MQTOpt dialect definition
MQTOpt = Dialect(
    "mqtopt",
    [
        MeasureOp,
        XOp,
        AllocOp,
        DeallocOp,
    ],
    [
        QubitType,
        QregType,
    ],
)
