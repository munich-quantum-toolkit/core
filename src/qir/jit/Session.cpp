/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/jit/Session.hpp"

#include "qir/jit/IRRewriter.hpp"
#include "qir/runtime/QIR.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/CoreContainers.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/LazyReexports.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mqt-core-qir-jit"

namespace qir {

static void exitOnLazyCallThroughFailure() { exit(1); }

static int mingwNoopMain() {
  // Cygwin and MinGW insert calls from the main function to the runtime
  // function __main. The __main function is responsible for setting up main's
  // environment (e.g. running static constructors), however this is not needed
  // when running under lli: the executor process will have run non-JIT ctors,
  // and ORC will take care of running JIT'd ctors. To avoid a missing symbol
  // error we just implement __main as a no-op.
  return 0;
}

// Try to enable debugger support for the given instance.
// This always returns success, but prints a warning if it's not able to enable
// debugger support.
static llvm::Error tryEnableDebugSupport(llvm::orc::LLJIT& jit) {
  if (auto err = enableDebuggerSupport(jit)) {
    [[maybe_unused]] const std::string errMsg = toString(std::move(err));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": " << errMsg << "\n");
  }
  return llvm::Error::success();
}

static llvm::Expected<llvm::orc::ThreadSafeModule>
getThreadSafeModuleOrError(std::unique_ptr<llvm::Module> module,
                           const llvm::SMDiagnostic& err,
                           llvm::orc::ThreadSafeContext tsCtx) {
  if (!module) {
    std::string errMsg;
    {
      llvm::raw_string_ostream errMsgStream(errMsg);
      err.print(DEBUG_TYPE, errMsgStream);
    }
    return llvm::make_error<llvm::StringError>(std::move(errMsg),
                                               llvm::inconvertibleErrorCode());
  }
  return llvm::orc::ThreadSafeModule(std::move(module), std::move(tsCtx));
}

llvm::Expected<llvm::orc::ThreadSafeModule>
JitSession::loadModuleFromFile(const llvm::StringRef irPath) {
  llvm::SMDiagnostic err;
  auto m = tsCtx_.withContextDo(
      [&](llvm::LLVMContext* ctx) { return parseIRFile(irPath, err, *ctx); });
  return getThreadSafeModuleOrError(std::move(m), err, tsCtx_);
}

llvm::Expected<llvm::orc::ThreadSafeModule>
JitSession::loadModuleFromMemory(const llvm::StringRef irBytes,
                                 const llvm::StringRef bufferName) {
  llvm::SMDiagnostic err;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(
      irBytes, bufferName,
      /*RequiresNullTerminator=*/false); // bitcode isn't null-terminated
  auto m = tsCtx_.withContextDo([&](llvm::LLVMContext* ctx) {
    return parseIR(buffer->getMemBufferRef(), err, *ctx);
  });
  return getThreadSafeModuleOrError(std::move(m), err, tsCtx_);
}

JitSession::JitSession(const llvm::StringRef inputFile, const Execution mode) {
  initialize(loadModuleFromFile(inputFile), mode);
}

JitSession::JitSession(const llvm::StringRef irBytes,
                       const llvm::StringRef bufferName, const Execution mode) {
  initialize(loadModuleFromMemory(irBytes, bufferName), mode);
}

JitSession::~JitSession() { deinitialize(); }

int JitSession::run(llvm::ArrayRef<std::string> args,
                    llvm::StringRef progName) const {
  // Manual in-process execution with RuntimeDyld.
  return llvm::orc::runAsMain(mainFn_, args, progName);
}

namespace {
std::vector<std::pair<std::string, void*>> manualSymbols;
} // namespace

#define REGISTER_SYMBOL(name)                                                  \
  llvm::sys::DynamicLibrary::AddSymbol(#name,                                  \
                                       reinterpret_cast<void*>(&(name)));      \
  manualSymbols.emplace_back(#name, reinterpret_cast<void*>(&(name)));

void JitSession::registerRuntimeSymbols() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    REGISTER_SYMBOL(__quantum__rt__result_get_zero);
    REGISTER_SYMBOL(__quantum__rt__result_get_one);
    REGISTER_SYMBOL(__quantum__rt__result_equal);
    REGISTER_SYMBOL(__quantum__rt__result_update_reference_count);
    REGISTER_SYMBOL(__quantum__rt__array_create_1d);
    REGISTER_SYMBOL(__quantum__rt__array_get_size_1d);
    REGISTER_SYMBOL(__quantum__rt__array_get_element_ptr_1d);
    REGISTER_SYMBOL(__quantum__rt__array_update_reference_count);
    REGISTER_SYMBOL(__quantum__rt__qubit_allocate);
    REGISTER_SYMBOL(__quantum__rt__qubit_allocate_array);
    REGISTER_SYMBOL(__quantum__rt__qubit_release);
    REGISTER_SYMBOL(__quantum__rt__qubit_release_array);
    REGISTER_SYMBOL(__quantum__qis__x__body);
    REGISTER_SYMBOL(__quantum__qis__y__body);
    REGISTER_SYMBOL(__quantum__qis__z__body);
    REGISTER_SYMBOL(__quantum__qis__h__body);
    REGISTER_SYMBOL(__quantum__qis__s__body);
    REGISTER_SYMBOL(__quantum__qis__sdg__body);
    REGISTER_SYMBOL(__quantum__qis__sx__body);
    REGISTER_SYMBOL(__quantum__qis__sxdg__body);
    REGISTER_SYMBOL(__quantum__qis__sqrtx__body);
    REGISTER_SYMBOL(__quantum__qis__sqrtxdg__body);
    REGISTER_SYMBOL(__quantum__qis__t__body);
    REGISTER_SYMBOL(__quantum__qis__tdg__body);
    REGISTER_SYMBOL(__quantum__qis__r__body);
    REGISTER_SYMBOL(__quantum__qis__prx__body);
    REGISTER_SYMBOL(__quantum__qis__rx__body);
    REGISTER_SYMBOL(__quantum__qis__ry__body);
    REGISTER_SYMBOL(__quantum__qis__rz__body);
    REGISTER_SYMBOL(__quantum__qis__p__body);
    REGISTER_SYMBOL(__quantum__qis__rxx__body);
    REGISTER_SYMBOL(__quantum__qis__ryy__body);
    REGISTER_SYMBOL(__quantum__qis__rzz__body);
    REGISTER_SYMBOL(__quantum__qis__rzx__body);
    REGISTER_SYMBOL(__quantum__qis__u__body);
    REGISTER_SYMBOL(__quantum__qis__u3__body);
    REGISTER_SYMBOL(__quantum__qis__u2__body);
    REGISTER_SYMBOL(__quantum__qis__u1__body);
    REGISTER_SYMBOL(__quantum__qis__cu1__body);
    REGISTER_SYMBOL(__quantum__qis__cu3__body);
    REGISTER_SYMBOL(__quantum__qis__cnot__body);
    REGISTER_SYMBOL(__quantum__qis__cx__body);
    REGISTER_SYMBOL(__quantum__qis__cy__body);
    REGISTER_SYMBOL(__quantum__qis__cz__body);
    REGISTER_SYMBOL(__quantum__qis__ch__body);
    REGISTER_SYMBOL(__quantum__qis__swap__body);
    REGISTER_SYMBOL(__quantum__qis__cswap__body);
    REGISTER_SYMBOL(__quantum__qis__crz__body);
    REGISTER_SYMBOL(__quantum__qis__cry__body);
    REGISTER_SYMBOL(__quantum__qis__crx__body);
    REGISTER_SYMBOL(__quantum__qis__cp__body);
    REGISTER_SYMBOL(__quantum__qis__ccx__body);
    REGISTER_SYMBOL(__quantum__qis__ccy__body);
    REGISTER_SYMBOL(__quantum__qis__ccz__body);
    REGISTER_SYMBOL(__quantum__qis__m__body);
    REGISTER_SYMBOL(__quantum__qis__measure__body);
    REGISTER_SYMBOL(__quantum__qis__mz__body);
    REGISTER_SYMBOL(__quantum__qis__reset__body);
    REGISTER_SYMBOL(__quantum__rt__initialize);
    REGISTER_SYMBOL(__quantum__rt__read_result);
    REGISTER_SYMBOL(__quantum__rt__result_record_output);
  });
}

#undef REGISTER_SYMBOL

void JitSession::initNativeTargets() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    static const llvm::codegen::RegisterCodeGenFlags CGF;

    // If we have a native target, initialize it to ensure it is linked in and
    // usable by the JIT.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}

void JitSession::initialize(
    llvm::Expected<llvm::orc::ThreadSafeModule> llvmModule,
    const Execution mode) {
  if (!llvmModule) {
    throw std::runtime_error(llvm::toString(llvmModule.takeError()));
  }
  module_ = std::move(*llvmModule);

  // In StateExtraction mode, strip QIR measurement and result-management calls
  // so the runtime's quantum state remains intact after main returns.
  if (mode == Execution::StateExtraction) {
    module_.withModuleDo(
        [](llvm::Module& m) { stripMeasurementRelatedCalls(m); });
  }

  registerRuntimeSymbols();
  initNativeTargets();

  // Get TargetTriple and DataLayout from the main module if they're explicitly
  // set.
  std::optional<llvm::Triple> tt;
  std::optional<llvm::DataLayout> dl;
  module_.withModuleDo([&](llvm::Module& m) {
    if (!m.getTargetTriple().empty()) {
      tt = m.getTargetTriple();
    }
    if (!m.getDataLayout().isDefault()) {
      dl = m.getDataLayout();
    }
  });

  // Configure the lazy JIT builder.
  llvm::orc::LLLazyJITBuilder builder;

  // Use the module's target triple if set, otherwise detect the host's.
  auto host = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!host) {
    throw std::runtime_error(llvm::toString(host.takeError()));
  }
  builder.setJITTargetMachineBuilder(
      tt ? llvm::orc::JITTargetMachineBuilder(*tt) : *host);

  // Cache the resolved triple; apply the module's explicit data layout if any.
  tt = builder.getJITTargetMachineBuilder()->getTargetTriple();
  if (dl) {
    builder.setDataLayout(dl);
  }

  // Optional architecture override from the -march codegen flag.
  if (!llvm::codegen::getMArch().empty()) {
    builder.getJITTargetMachineBuilder()->getTargetTriple().setArchName(
        llvm::codegen::getMArch());
  }

  // Apply CPU, features, relocation model, and code model from codegen flags.
  builder.getJITTargetMachineBuilder()
      ->setCPU(llvm::codegen::getCPUStr())
      .addFeatures(llvm::codegen::getFeatureList())
      .setRelocationModel(llvm::codegen::getExplicitRelocModel())
      .setCodeModel(llvm::codegen::getExplicitCodeModel());

  // Link process symbols.
  builder.setLinkProcessSymbolsByDefault(true);

  // Set up the in-process execution session and lazy call-through manager.
  auto pc = llvm::orc::SelfExecutorProcessControl::Create();
  if (!pc) {
    throw std::runtime_error(llvm::toString(pc.takeError()));
  }
  auto es = std::make_unique<llvm::orc::ExecutionSession>(std::move(*pc));
  builder.setLazyCallthroughManager(
      std::make_unique<llvm::orc::LazyCallThroughManager>(
          *es, llvm::orc::ExecutorAddr(), nullptr));
  builder.setExecutionSession(std::move(es));

  // Abort on lazy compilation failure.
  builder.setLazyCompileFailureAddr(
      llvm::orc::ExecutorAddr::fromPtr(exitOnLazyCallThroughFailure));

  // Enable debugging of JIT'd code (only works on JITLink for ELF and MachO).
  builder.setPrePlatformSetup(tryEnableDebugSupport);

  // Build the JIT.
  auto expectedJit = builder.create();
  if (!expectedJit) {
    throw std::runtime_error(llvm::toString(expectedJit.takeError()));
  }
  jit_ = std::move(*expectedJit);

  // Register QIR runtime symbols.
  auto& jd = jit_->getMainJITDylib();
  llvm::orc::SymbolMap hostSymbols;
  for (const auto& [name, ptr] : manualSymbols) {
    hostSymbols[jit_->mangleAndIntern(name)] = {
        llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported};
  }
  if (auto err = jd.define(llvm::orc::absoluteSymbols(hostSymbols))) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }

  // DynamicLibrarySearchGenerator
  auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      jit_->getDataLayout().getGlobalPrefix());
  if (!gen) {
    throw std::runtime_error(llvm::toString(gen.takeError()));
  }
  jit_->getMainJITDylib().addGenerator(std::move(*gen));

  // GDB listener (no error path)
  auto* objLayer = &jit_->getObjLinkingLayer();
  if (auto* rtDyldObjLayer =
          dyn_cast<llvm::orc::RTDyldObjectLinkingLayer>(objLayer)) {
    rtDyldObjLayer->registerJITEventListener(
        *llvm::JITEventListener::createGDBRegistrationListener());
  }

  // If this is a Mingw or Cygwin executor then we need to alias __main to
  // orc_rt_int_void_return_0.
  if (jit_->getTargetTriple().isOSCygMing()) {
    auto& workaroundJD = jit_->getProcessSymbolsJITDylib()
                             ? *jit_->getProcessSymbolsJITDylib()
                             : jit_->getMainJITDylib();
    if (auto err = workaroundJD.define(llvm::orc::absoluteSymbols(
            {{jit_->mangleAndIntern("__main"),
              {llvm::orc::ExecutorAddr::fromPtr(mingwNoopMain),
               llvm::JITSymbolFlags::Exported}}}))) {
      throw std::runtime_error(llvm::toString(std::move(err)));
    }
  }

  // Regular modules are greedy: They materialize as a whole and trigger
  // materialization for all required symbols recursively. Lazy modules go
  // through partitioning, and they replace outgoing calls with reexport stubs
  // that resolve on call-through.
  auto addModule = [&](llvm::orc::JITDylib& jdlib,
                       llvm::orc::ThreadSafeModule m) {
    return jit_->addIRModule(jdlib, std::move(m));
  };

  // Add the main module.
  if (auto err = addModule(jit_->getMainJITDylib(), std::move(module_))) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }

  // Run any static constructors.
  if (auto err = jit_->initialize(jit_->getMainJITDylib())) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }

  // Resolve the main function.
  auto mainAddr = jit_->lookup("main");
  if (!mainAddr) {
    throw std::runtime_error(llvm::toString(mainAddr.takeError()));
  }
  mainFn_ = mainAddr->toPtr<MainFn*>();
}

void JitSession::deinitialize() const {
  if (!jit_) {
    return;
  }
  if (auto err = jit_->deinitialize(jit_->getMainJITDylib())) {
    llvm::errs() << "JitSession deinitialize failed: "
                 << llvm::toString(std::move(err)) << "\n";
  }
}

} // namespace qir
