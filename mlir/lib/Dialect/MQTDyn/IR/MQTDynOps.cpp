#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTDyn/IR/MQTDynOpsDialect.cpp.inc"

void mqt::ir::dyn::MQTDynDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MQTDyn/IR/MQTDynOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MQTDyn/IR/MQTDynOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MQTDyn/IR/MQTDynOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MQTDyn/IR/MQTDynInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MQTDyn/IR/MQTDynOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifier
//===----------------------------------------------------------------------===//

namespace mqt::ir::dyn {

mlir::LogicalResult MeasureOp::verify() {
  if (getInQubits().size() != getOutBits().size()) {
    return emitOpError() << "number of input qubits (" << getInQubits().size()
                         << ") " << "and output bits (" << getOutBits().size()
                         << ") must be the same";
  }
  return mlir::success();
}

mlir::LogicalResult AllocOp::verify() {
  if (!getSize() && !getSizeAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for size";
  }
  if (getSize() && getSizeAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for size";
  }
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  if (!getIndex() && !getIndexAttr().has_value()) {
    return emitOpError() << "expected an operand or attribute for index";
  }
  if (getIndex() && getIndexAttr().has_value()) {
    return emitOpError() << "expected either an operand or attribute for index";
  }
  return mlir::success();
}

} // namespace mqt::ir::dyn