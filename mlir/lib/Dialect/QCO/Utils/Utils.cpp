#include "mlir/Dialect/QCO/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qco {
func::FuncOp getEntryPoint(ModuleOp op) {
  static constexpr llvm::StringRef PASSTHROUGH_LABEL = "passthrough";
  static constexpr llvm::StringRef ENTRY_POINT_LABEL = "entry_point";

  const auto isEntry = [](Attribute attr) {
    const auto strAttr = dyn_cast<StringAttr>(attr);
    return strAttr && strAttr.getValue() == ENTRY_POINT_LABEL;
  };

  for (auto func : op.getOps<func::FuncOp>()) {
    const auto passthrough = func->getAttrOfType<ArrayAttr>(PASSTHROUGH_LABEL);
    if (!passthrough) {
      continue;
    }

    if (llvm::any_of(passthrough, isEntry)) {
      return func;
    }
  }

  return nullptr;
}
} // namespace mlir::qco