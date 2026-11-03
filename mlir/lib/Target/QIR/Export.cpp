/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Target/QIR/Export.h"

#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <system_error>

namespace mlir::qir {

static LogicalResult materializeNativeQISNames(ModuleOp moduleOp) {
  SymbolTable symbols(moduleOp);
  const auto result = moduleOp.walk([&](LLVM::CallOp call) {
    auto name = call->getAttrOfType<StringAttr>(qir::QIR_NATIVE_NAME_ATTR);
    if (!name) {
      if (call.getCallee() &&
          call.getCallee()->starts_with("__quantum__qis__")) {
        call.emitError("target QIR contains an unchecked quantum instruction");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (!call.getCallee()) {
      call.emitError("native QIR requires direct quantum instruction calls");
      return WalkResult::interrupt();
    }
    auto source = symbols.lookup<LLVM::LLVMFuncOp>(*call.getCallee());
    auto destination = symbols.lookup<LLVM::LLVMFuncOp>(name.getValue());
    if (destination &&
        destination.getFunctionType() != source.getFunctionType()) {
      call.emitError("native QIR operations require conflicting signatures");
      return WalkResult::interrupt();
    }
    if (!destination) {
      destination = cast<LLVM::LLVMFuncOp>(source->clone());
      destination.setName(name.getValue());
      symbols.insert(destination);
    }
    call.setCalleeAttr(FlatSymbolRefAttr::get(name));
    call->removeAttr(qir::QIR_NATIVE_NAME_ATTR);
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  const auto uses = SymbolTable::getSymbolUses(&moduleOp.getBodyRegion());
  if (!uses) {
    return success();
  }
  DenseSet<StringAttr> referencedSymbols;
  for (const auto& use : *uses) {
    referencedSymbols.insert(use.getSymbolRef().getRootReference());
  }
  for (auto function :
       llvm::make_early_inc_range(moduleOp.getOps<LLVM::LLVMFuncOp>())) {
    if (function.isExternal() &&
        function.getName().starts_with("__quantum__qis__") &&
        !referencedSymbols.contains(function.getSymNameAttr())) {
      function.erase();
    }
  }
  return success();
}

std::unique_ptr<llvm::Module> translateQIRToLLVM(ModuleOp mod,
                                                 llvm::LLVMContext& context) {
  auto llvmModule = translateModuleToLLVMIR(mod, context);
  if (!llvmModule) {
    mod.emitError("failed to translate QIR MLIR to LLVM IR");
    return nullptr;
  }
  qir::normalizeQIRModuleFlags(*llvmModule);
  if (llvm::verifyModule(*llvmModule, &llvm::errs())) {
    mod.emitError("exported QIR failed LLVM IR verification");
    return nullptr;
  }
  return llvmModule;
}

[[nodiscard]] static std::unique_ptr<llvm::Module>
translateToLLVM(ModuleOp mod, llvm::LLVMContext& context) {
  auto environment =
      mod->getAttrOfType<mqt::TargetEnvAttr>(mqt::TargetEnvAttr::name);
  if (environment && environment.getCompilationTarget().getNativeOperations() ==
                         mqt::NativeOperationsKind::Explicit) {
    if (failed(materializeNativeQISNames(mod))) {
      return nullptr;
    }
  }
  return translateQIRToLLVM(mod, context);
}

LogicalResult translateQIRToLLVMIR(OwningOpRef<ModuleOp> moduleOp,
                                   llvm::raw_ostream& output) {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(*moduleOp, context);
  if (!llvmModule) {
    return failure();
  }
  llvmModule->print(output, nullptr);
  return success();
}

LogicalResult translateQIRToBitcode(OwningOpRef<ModuleOp> moduleOp,
                                    llvm::raw_ostream& output) {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(*moduleOp, context);
  if (!llvmModule) {
    return failure();
  }
  llvm::WriteBitcodeToFile(*llvmModule, output);
  return success();
}

LogicalResult writeQIRBitcode(OwningOpRef<ModuleOp> moduleOp,
                              StringRef filename) {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(*moduleOp, context);
  if (!llvmModule) {
    return failure();
  }

  std::error_code error;
  llvm::raw_fd_ostream stream(filename, error, llvm::sys::fs::OF_None);
  if (error) {
    moduleOp->emitError() << "failed to open bitcode output file '" << filename
                          << "': " << error.message();
    return failure();
  }
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  stream.flush();
  if (stream.has_error()) {
    stream.clear_error();
    moduleOp->emitError() << "failed to write bitcode file '" << filename
                          << "'";
    return failure();
  }
  return success();
}

} // namespace mlir::qir
