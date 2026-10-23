/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Target/IQM/Export.h"

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QIR/QIRDefinitions.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "support/JSON.hpp"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir::qir {

static llvm::Error fail(const Twine& message) {
  return llvm::createStringError(Twine("IQM JSON export: ") + message);
}

static std::optional<uint64_t> staticIndex(const llvm::Value* value) {
  if (llvm::isa<llvm::ConstantPointerNull>(value)) {
    return 0;
  }
  const auto* cast = llvm::dyn_cast<llvm::ConstantExpr>(value);
  if (cast == nullptr || cast->getOpcode() != llvm::Instruction::IntToPtr) {
    return std::nullopt;
  }
  const auto* index = llvm::dyn_cast<llvm::ConstantInt>(cast->getOperand(0));
  if (index == nullptr || index->isNegative() ||
      index->getValue().getActiveBits() > 64) {
    return std::nullopt;
  }
  return index->getZExtValue();
}

static std::optional<double> parameter(const llvm::Value* value) {
  const auto* constant = llvm::dyn_cast<llvm::ConstantFP>(value);
  if (constant == nullptr || !constant->getType()->isDoubleTy()) {
    return std::nullopt;
  }
  const auto number = constant->getValueAPF().convertToDouble();
  return std::isfinite(number) ? std::optional{number} : std::nullopt;
}

namespace {

class Exporter {
public:
  Exporter(const llvm::Module& module, const CompilerTarget& target)
      : module_(module), target_(target) {}

  llvm::Expected<std::string> run() {
    if (llvm::verifyModule(module_, &llvm::errs())) {
      return fail("invalid LLVM module");
    }
    const llvm::Function* entry = nullptr;
    for (const auto& function : module_) {
      if (!function.hasFnAttribute(::qir::ENTRY_POINT_ATTR)) {
        continue;
      }
      if (entry != nullptr) {
        return fail("expected exactly one QIR entry point");
      }
      entry = &function;
    }
    if (entry == nullptr || entry->isDeclaration() || !entry->arg_empty() ||
        !entry->getFnAttribute(::qir::QIR_PROFILES_ATTR).isStringAttribute() ||
        entry->getFnAttribute(::qir::QIR_PROFILES_ATTR).getValueAsString() !=
            StringRef(::qir::BASE_PROFILE)) {
      return fail("expected a defined QIR Base entry point without arguments");
    }
    if (entry->getName().empty() || !llvm::json::isUTF8(entry->getName())) {
      return fail("expected a nonempty UTF-8 entry point name");
    }
    llvm::StringSet<> names;
    for (const auto& site : target_.sites()) {
      const auto name = site.name();
      if (!name || name->empty() || !llvm::json::isUTF8(*name) ||
          !names.insert(*name).second) {
        return fail("target site names must be nonempty, unique UTF-8 strings");
      }
      siteNames_.try_emplace(static_cast<uint64_t>(site.id()), *name);
    }

    std::string output;
    llvm::raw_string_ostream raw(output);
    llvm::json::OStream stream(raw);
    DenseSet<const llvm::BasicBlock*> visited;
    stream.objectBegin();
    stream.attribute("name", entry->getName());
    stream.attributeBegin("instructions");
    stream.arrayBegin();
    auto error = [&]() -> llvm::Error {
      const auto* block = &entry->getEntryBlock();
      while (block != nullptr) {
        if (!visited.insert(block).second) {
          return fail("control flow must be an acyclic unconditional chain");
        }
        const llvm::BasicBlock* next = nullptr;
        for (const auto& instruction : *block) {
          if (const auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
            if (auto error = emitCall(*call, stream)) {
              return std::move(error);
            }
          } else if (const auto* branch =
                         llvm::dyn_cast<llvm::BranchInst>(&instruction);
                     branch != nullptr && branch->isUnconditional()) {
            next = branch->getSuccessor(0);
          } else if (const auto* result =
                         llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
            const auto* value = result->getReturnValue();
            const auto* integer =
                llvm::dyn_cast_or_null<llvm::ConstantInt>(value);
            if (value != nullptr &&
                (integer == nullptr || !integer->isZero())) {
              return fail("entry point may only return a zero exit code");
            }
          } else {
            return fail("unsupported LLVM instruction");
          }
        }
        block = next;
      }
      return llvm::Error::success();
    }();
    stream.arrayEnd();
    stream.attributeEnd();
    stream.objectEnd();
    if (error) {
      return std::move(error);
    }
    if (visited.size() != entry->size() || pendingArrayElements_ != 0 ||
        measuredQubits_.empty() || !measurements_.empty()) {
      return fail("every measurement must be recorded exactly once in a "
                  "complete output sequence");
    }
    return output;
  }

private:
  const llvm::Module& module_;
  const CompilerTarget& target_;
  DenseMap<uint64_t, StringRef> siteNames_;
  DenseSet<uint64_t> measuredQubits_;
  DenseMap<uint64_t, uint64_t> measurements_;
  uint64_t pendingArrayElements_ = 0;
  uint64_t registerSize_ = 0;
  uint64_t registerCount_ = 0;
  std::string registerName_;
  bool hasInstructions_ = false;

  llvm::Error beginRegister(const llvm::Value* label, const uint64_t size) {
    if (llvm::isa<llvm::ConstantPointerNull>(label)) {
      registerName_ = "c" + std::to_string(registerCount_);
    } else {
      StringRef name;
      if (!llvm::getConstantStringInfo(label, name, false) ||
          !name.contains('\0') || !llvm::json::isUTF8(name.split('\0').first)) {
        return fail(
            "output labels must be constant null-terminated UTF-8 strings");
      }
      registerName_ = name.split('\0').first.str();
    }
    registerSize_ = pendingArrayElements_ = size;
    ++registerCount_;
    return llvm::Error::success();
  }

  llvm::Error emitCall(const llvm::CallInst& call,
                       llvm::json::OStream& stream) {
    const auto* function = call.getCalledFunction();
    if (function == nullptr || !function->isDeclaration() ||
        !call.getType()->isVoidTy()) {
      return fail("only direct QIR calls returning void are supported");
    }
    const auto name = function->getName();
    const auto arity = call.arg_size();
    if (name == QIR_INITIALIZE && arity == 1 &&
        llvm::isa<llvm::ConstantPointerNull>(call.getArgOperand(0)) &&
        !hasInstructions_ && measuredQubits_.empty()) {
      return llvm::Error::success();
    }
    if (name == QIR_ARRAY_RECORD_OUTPUT && arity == 2) {
      const auto* count =
          llvm::dyn_cast<llvm::ConstantInt>(call.getArgOperand(0));
      if (pendingArrayElements_ != 0 || count == nullptr ||
          count->isNegative() || count->getValue().getActiveBits() > 64) {
        return fail("output arrays must contain a constant number of results");
      }
      return beginRegister(call.getArgOperand(1), count->getZExtValue());
    }
    if (name == QIR_RECORD_OUTPUT && arity == 2) {
      const auto result = staticIndex(call.getArgOperand(0));
      if (!result || !measurements_.contains(*result)) {
        return fail("output must record a unique measured result handle");
      }
      if (pendingArrayElements_ == 0) {
        if (auto error = beginRegister(call.getArgOperand(1), 1)) {
          return error;
        }
      }
      const auto key = (Twine(registerName_) + "_" + Twine(registerSize_) +
                        "_" + Twine(registerCount_ - 1) + "_" +
                        Twine(registerSize_ - pendingArrayElements_))
                           .str();
      stream.object([&] {
        stream.attribute("name", "measure");
        stream.attributeArray("locus", [&] {
          stream.value(siteNames_.at(measurements_.at(*result)));
        });
        stream.attributeObject("args", [&] { stream.attribute("key", key); });
      });
      hasInstructions_ = true;
      measurements_.erase(*result);
      --pendingArrayElements_;
      return llvm::Error::success();
    }
    if (name == QIR_MEASURE && arity == 2) {
      const auto qubit = staticIndex(call.getArgOperand(0));
      const auto result = staticIndex(call.getArgOperand(1));
      if (!qubit || !result || !siteNames_.contains(*qubit) ||
          !measuredQubits_.insert(*qubit).second ||
          !measurements_.try_emplace(*result, *qubit).second ||
          registerCount_ != 0) {
        return fail("measurements require unique target qubits and result "
                    "handles before output recording");
      }
      return llvm::Error::success();
    }
    if (!measuredQubits_.empty()) {
      return fail("gates must precede all measurements");
    }
    if (name == QIR_R && arity == 3) {
      const auto angle = parameter(call.getArgOperand(0));
      const auto phase = parameter(call.getArgOperand(1));
      const auto qubit = staticIndex(call.getArgOperand(2));
      if (!angle || !phase || !qubit || !siteNames_.contains(*qubit)) {
        return fail("PRX requires finite constant angles and a target qubit");
      }
      stream.object([&] {
        stream.attribute("name", "prx");
        stream.attributeArray("locus",
                              [&] { stream.value(siteNames_.at(*qubit)); });
        stream.attributeObject("args", [&] {
          stream.attributeBegin("angle");
          ::mqt::json::number(stream, *angle);
          stream.attributeEnd();
          stream.attributeBegin("phase");
          ::mqt::json::number(stream, *phase);
          stream.attributeEnd();
        });
      });
      hasInstructions_ = true;
      return llvm::Error::success();
    }
    if (name == QIR_CZ && arity == 2) {
      const auto control = staticIndex(call.getArgOperand(0));
      const auto qubit = staticIndex(call.getArgOperand(1));
      if (!control || !qubit || *control == *qubit ||
          !siteNames_.contains(*control) || !siteNames_.contains(*qubit)) {
        return fail("CZ requires two distinct target qubits");
      }
      stream.object([&] {
        stream.attribute("name", "cz");
        stream.attributeArray("locus", [&] {
          stream.value(siteNames_.at(*control));
          stream.value(siteNames_.at(*qubit));
        });
        stream.attributeObject("args", [] {});
      });
      hasInstructions_ = true;
      return llvm::Error::success();
    }
    return fail(Twine("unsupported call '") + name + "'");
  }
};

} // namespace

llvm::Expected<std::string> exportIQMJSON(const llvm::Module& module,
                                          const CompilerTarget& target) {
  return Exporter(module, target).run();
}

} // namespace mlir::qir
