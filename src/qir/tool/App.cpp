/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/backend/QIR.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/DebugUtils.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/LazyReexports.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/SymbolStringPool.h>
#include <llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "mqt-core-qir-runner"

namespace {
llvm::cl::opt<std::string> InputFile(llvm::cl::desc("<input bitcode>"),
                                     llvm::cl::Positional, llvm::cl::init("-"));

llvm::cl::list<std::string> InputArgv(llvm::cl::ConsumeAfter,
                                      llvm::cl::desc("<program arguments>..."));

llvm::ExitOnError ExitOnErr;

static void exitOnLazyCallThroughFailure() { exit(1); }

static std::function<void(llvm::Module&)> createIRDebugDumper() {
  return [](llvm::Module& M) {};
}

static std::function<void(llvm::MemoryBuffer&)> createObjDebugDumper() {
  return [](llvm::MemoryBuffer&) {};
}

llvm::Expected<llvm::orc::ThreadSafeModule>
loadModule(llvm::StringRef Path, llvm::orc::ThreadSafeContext TSCtx) {
  llvm::SMDiagnostic Err;
  auto M = TSCtx.withContextDo(
      [&](llvm::LLVMContext* Ctx) { return parseIRFile(Path, Err, *Ctx); });
  if (!M) {
    std::string ErrMsg;
    {
      llvm::raw_string_ostream ErrMsgStream(ErrMsg);
      Err.print("lli", ErrMsgStream);
    }
    return llvm::make_error<llvm::StringError>(std::move(ErrMsg),
                                               llvm::inconvertibleErrorCode());
  }

  return llvm::orc::ThreadSafeModule(std::move(M), std::move(TSCtx));
}

int mingw_noop_main(void) {
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
llvm::Error tryEnableDebugSupport(llvm::orc::LLJIT& J) {
  if (auto Err = enableDebuggerSupport(J)) {
    [[maybe_unused]] std::string ErrMsg = toString(std::move(Err));
    LLVM_DEBUG(llvm::dbgs() << "lli: " << ErrMsg << "\n");
  }
  return llvm::Error::success();
}

int runOrcJIT(const char* ProgName) {
  // Start setting up the JIT environment.

  // Parse the main module.
  llvm::orc::ThreadSafeContext TSCtx(std::make_unique<llvm::LLVMContext>());
  auto MainModule = ExitOnErr(loadModule(InputFile, TSCtx));

  // Get TargetTriple and DataLayout from the main module if they're explicitly
  // set.
  std::optional<llvm::Triple> TT;
  std::optional<llvm::DataLayout> DL;
  MainModule.withModuleDo([&](llvm::Module& M) {
    if (!M.getTargetTriple().empty())
      TT = M.getTargetTriple();
    if (!M.getDataLayout().isDefault())
      DL = M.getDataLayout();
  });

  llvm::orc::LLLazyJITBuilder Builder;

  Builder.setJITTargetMachineBuilder(
      TT ? llvm::orc::JITTargetMachineBuilder(*TT)
         : ExitOnErr(llvm::orc::JITTargetMachineBuilder::detectHost()));

  TT = Builder.getJITTargetMachineBuilder()->getTargetTriple();
  if (DL) {
    Builder.setDataLayout(DL);
  }

  if (!llvm::codegen::getMArch().empty()) {
    Builder.getJITTargetMachineBuilder()->getTargetTriple().setArchName(
        llvm::codegen::getMArch());
  }

  Builder.getJITTargetMachineBuilder()
      ->setCPU(llvm::codegen::getCPUStr())
      .addFeatures(llvm::codegen::getFeatureList())
      .setRelocationModel(llvm::codegen::getExplicitRelocModel())
      .setCodeModel(llvm::codegen::getExplicitCodeModel());

  // Link process symbols.
  Builder.setLinkProcessSymbolsByDefault(true);

  auto ES = std::make_unique<llvm::orc::ExecutionSession>(
      ExitOnErr(llvm::orc::SelfExecutorProcessControl::Create()));
  Builder.setLazyCallthroughManager(
      std::make_unique<llvm::orc::LazyCallThroughManager>(
          *ES, llvm::orc::ExecutorAddr(), nullptr));
  Builder.setExecutionSession(std::move(ES));

  Builder.setLazyCompileFailureAddr(
      llvm::orc::ExecutorAddr::fromPtr(exitOnLazyCallThroughFailure));

  // Enable debugging of JIT'd code (only works on JITLink for ELF and MachO).
  Builder.setPrePlatformSetup(tryEnableDebugSupport);

  auto J = ExitOnErr(Builder.create());

  auto* ObjLayer = &J->getObjLinkingLayer();
  if (auto* RTDyldObjLayer =
          dyn_cast<llvm::orc::RTDyldObjectLinkingLayer>(ObjLayer)) {
    RTDyldObjLayer->registerJITEventListener(
        *llvm::JITEventListener::createGDBRegistrationListener());
  }

  auto IRDump = createIRDebugDumper();
  J->getIRTransformLayer().setTransform(
      [&](llvm::orc::ThreadSafeModule TSM,
          const llvm::orc::MaterializationResponsibility& R) {
        TSM.withModuleDo([&](llvm::Module& M) {
          if (verifyModule(M, &llvm::dbgs())) {
            llvm::dbgs() << "Bad module: " << &M << "\n";
            exit(1);
          }
          IRDump(M);
        });
        return TSM;
      });

  auto ObjDump = createObjDebugDumper();
  J->getObjTransformLayer().setTransform(
      [&](std::unique_ptr<llvm::MemoryBuffer> Obj)
          -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
        ObjDump(*Obj);
        return std::move(Obj);
      });

  // If this is a Mingw or Cygwin executor then we need to alias __main to
  // orc_rt_int_void_return_0.
  if (J->getTargetTriple().isOSCygMing()) {
    auto& WorkaroundJD = J->getProcessSymbolsJITDylib()
                             ? *J->getProcessSymbolsJITDylib()
                             : J->getMainJITDylib();
    ExitOnErr(WorkaroundJD.define(llvm::orc::absoluteSymbols(
        {{J->mangleAndIntern("__main"),
          {llvm::orc::ExecutorAddr::fromPtr(mingw_noop_main),
           llvm::JITSymbolFlags::Exported}}})));
  }

  // Regular modules are greedy: They materialize as a whole and trigger
  // materialization for all required symbols recursively. Lazy modules go
  // through partitioning and they replace outgoing calls with reexport stubs
  // that resolve on call-through.
  auto AddModule = [&](llvm::orc::JITDylib& JD, llvm::orc::ThreadSafeModule M) {
    return J->addIRModule(JD, std::move(M));
  };

  // Add the main module.
  ExitOnErr(AddModule(J->getMainJITDylib(), std::move(MainModule)));

  // Run any static constructors.
  ExitOnErr(J->initialize(J->getMainJITDylib()));

  // Resolve and run the main function.
  auto MainAddr = ExitOnErr(J->lookup("main"));

  // Manual in-process execution with RuntimeDyld.
  using MainFnTy = int(int, char*[]);
  auto MainFn = MainAddr.toPtr<MainFnTy*>();
  int Result =
      llvm::orc::runAsMain(MainFn, InputArgv, llvm::StringRef(InputFile));

  // Run destructors.
  ExitOnErr(J->deinitialize(J->getMainJITDylib()));

  return Result;
}
} // namespace

auto main(int argc, char* argv[]) -> int {
  const llvm::InitLLVM X(argc, argv);

  if (argc > 1) {
    ExitOnErr.setBanner(std::string(argv[0]) + ": ");
  }

  // If we have a native target, initialize it to ensure it is linked in and
  // usable by the JIT.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qir interpreter & dynamic compiler\n");

  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__result_get_zero",
                                       (void*)&__quantum__rt__result_get_zero);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__result_get_one",
                                       (void*)&__quantum__rt__result_get_one);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__result_equal",
                                       (void*)&__quantum__rt__result_equal);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_update_reference_count",
      (void*)&__quantum__rt__result_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__string_create",
                                       (void*)&__quantum__rt__string_create);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__string_get_data",
                                       (void*)&__quantum__rt__string_get_data);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_get_length",
      (void*)&__quantum__rt__string_get_length);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_update_reference_count",
      (void*)&__quantum__rt__string_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_concatenate",
      (void*)&__quantum__rt__string_concatenate);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__string_equal",
                                       (void*)&__quantum__rt__string_equal);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__int_to_string",
                                       (void*)&__quantum__rt__int_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__double_to_string",
                                       (void*)&__quantum__rt__double_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bool_to_string",
                                       (void*)&__quantum__rt__bool_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__result_to_string",
                                       (void*)&__quantum__rt__result_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__pauli_to_string",
                                       (void*)&__quantum__rt__pauli_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__qubit_to_string",
                                       (void*)&__quantum__rt__qubit_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__range_to_string",
                                       (void*)&__quantum__rt__range_to_string);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_to_string",
                                       (void*)&__quantum__rt__bigint_to_string);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_create_i64",
      (void*)&__quantum__rt__bigint_create_i64);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_create_array",
      (void*)&__quantum__rt__bigint_create_array);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_get_data",
                                       (void*)&__quantum__rt__bigint_get_data);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_get_length",
      (void*)&__quantum__rt__bigint_get_length);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_update_reference_count",
      (void*)&__quantum__rt__bigint_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_negate",
                                       (void*)&__quantum__rt__bigint_negate);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_add",
                                       (void*)&__quantum__rt__bigint_add);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_subtract",
                                       (void*)&__quantum__rt__bigint_subtract);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_multiply",
                                       (void*)&__quantum__rt__bigint_multiply);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_divide",
                                       (void*)&__quantum__rt__bigint_divide);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_modulus",
                                       (void*)&__quantum__rt__bigint_modulus);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_power",
                                       (void*)&__quantum__rt__bigint_power);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_bitand",
                                       (void*)&__quantum__rt__bigint_bitand);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_bitor",
                                       (void*)&__quantum__rt__bigint_bitor);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_bitxor",
                                       (void*)&__quantum__rt__bigint_bitxor);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_bitnot",
                                       (void*)&__quantum__rt__bigint_bitnot);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_shiftleft",
                                       (void*)&__quantum__rt__bigint_shiftleft);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_shiftright",
      (void*)&__quantum__rt__bigint_shiftright);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_equal",
                                       (void*)&__quantum__rt__bigint_equal);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__bigint_greater",
                                       (void*)&__quantum__rt__bigint_greater);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_greater_eq",
      (void*)&__quantum__rt__bigint_greater_eq);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__tuple_create",
                                       (void*)&__quantum__rt__tuple_create);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__tuple_copy",
                                       (void*)&__quantum__rt__tuple_copy);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_update_reference_count",
      (void*)&__quantum__rt__tuple_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_update_alias_count",
      (void*)&__quantum__rt__tuple_update_alias_count);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_create_1d",
                                       (void*)&__quantum__rt__array_create_1d);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_copy",
                                       (void*)&__quantum__rt__array_copy);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_concatenate",
      (void*)&__quantum__rt__array_concatenate);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_slice_1d",
                                       (void*)&__quantum__rt__array_slice_1d);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_size_1d",
      (void*)&__quantum__rt__array_get_size_1d);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_element_ptr_1d",
      (void*)&__quantum__rt__array_get_element_ptr_1d);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_update_reference_count",
      (void*)&__quantum__rt__array_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_update_alias_count",
      (void*)&__quantum__rt__array_update_alias_count);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_create",
                                       (void*)&__quantum__rt__array_create);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_get_dim",
                                       (void*)&__quantum__rt__array_get_dim);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_get_size",
                                       (void*)&__quantum__rt__array_get_size);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_element_ptr",
      (void*)&__quantum__rt__array_get_element_ptr);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_slice",
                                       (void*)&__quantum__rt__array_slice);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__array_project",
                                       (void*)&__quantum__rt__array_project);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__callable_create",
                                       (void*)&__quantum__rt__callable_create);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__callable_copy",
                                       (void*)&__quantum__rt__callable_copy);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__callable_invoke",
                                       (void*)&__quantum__rt__callable_invoke);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_make_adjoint",
      (void*)&__quantum__rt__callable_make_adjoint);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_make_controlled",
      (void*)&__quantum__rt__callable_make_controlled);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_update_reference_count",
      (void*)&__quantum__rt__callable_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_update_alias_count",
      (void*)&__quantum__rt__callable_update_alias_count);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__capture_update_reference_count",
      (void*)&__quantum__rt__capture_update_reference_count);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__capture_update_alias_count",
      (void*)&__quantum__rt__capture_update_alias_count);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__message",
                                       (void*)&__quantum__rt__message);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__fail",
                                       (void*)&__quantum__rt__fail);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__qubit_allocate",
                                       (void*)&__quantum__rt__qubit_allocate);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_allocate_array",
      (void*)&__quantum__rt__qubit_allocate_array);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__qubit_release",
                                       (void*)&__quantum__rt__qubit_release);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_release_array",
      (void*)&__quantum__rt__qubit_release_array);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__x__body",
                                       (void*)&__quantum__qis__x__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__y__body",
                                       (void*)&__quantum__qis__y__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__z__body",
                                       (void*)&__quantum__qis__z__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__h__body",
                                       (void*)&__quantum__qis__h__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__s__body",
                                       (void*)&__quantum__qis__s__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__sdg__body",
                                       (void*)&__quantum__qis__sdg__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__sx__body",
                                       (void*)&__quantum__qis__sx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__sxdg__body",
                                       (void*)&__quantum__qis__sxdg__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__sqrtx__body",
                                       (void*)&__quantum__qis__sqrtx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__sqrtxdg__body",
                                       (void*)&__quantum__qis__sqrtxdg__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__t__body",
                                       (void*)&__quantum__qis__t__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__tdg__body",
                                       (void*)&__quantum__qis__tdg__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__rx__body",
                                       (void*)&__quantum__qis__rx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__ry__body",
                                       (void*)&__quantum__qis__ry__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__rz__body",
                                       (void*)&__quantum__qis__rz__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__p__body",
                                       (void*)&__quantum__qis__p__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__rxx__body",
                                       (void*)&__quantum__qis__rxx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__u__body",
                                       (void*)&__quantum__qis__u__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__u3__body",
                                       (void*)&__quantum__qis__u3__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__u2__body",
                                       (void*)&__quantum__qis__u2__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__u1__body",
                                       (void*)&__quantum__qis__u1__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cu1__body",
                                       (void*)&__quantum__qis__cu1__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cu3__body",
                                       (void*)&__quantum__qis__cu3__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cnot__body",
                                       (void*)&__quantum__qis__cnot__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cx__body",
                                       (void*)&__quantum__qis__cx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cy__body",
                                       (void*)&__quantum__qis__cy__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cz__body",
                                       (void*)&__quantum__qis__cz__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__ch__body",
                                       (void*)&__quantum__qis__ch__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__swap__body",
                                       (void*)&__quantum__qis__swap__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cswap__body",
                                       (void*)&__quantum__qis__cswap__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__crz__body",
                                       (void*)&__quantum__qis__crz__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cry__body",
                                       (void*)&__quantum__qis__cry__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__crx__body",
                                       (void*)&__quantum__qis__crx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__cp__body",
                                       (void*)&__quantum__qis__cp__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__rzz__body",
                                       (void*)&__quantum__qis__rzz__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__ccx__body",
                                       (void*)&__quantum__qis__ccx__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__ccz__body",
                                       (void*)&__quantum__qis__ccz__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__m__body",
                                       (void*)&__quantum__qis__m__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__measure__body",
                                       (void*)&__quantum__qis__measure__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__mz__body",
                                       (void*)&__quantum__qis__mz__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__qis__reset__body",
                                       (void*)&__quantum__qis__reset__body);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__initialize",
                                       (void*)&__quantum__rt__initialize);
  llvm::sys::DynamicLibrary::AddSymbol("__quantum__rt__read_result",
                                       (void*)&__quantum__rt__read_result);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_record_output",
      (void*)&__quantum__rt__tuple_record_output);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_record_output",
      (void*)&__quantum__rt__array_record_output);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_record_output",
      (void*)&__quantum__rt__result_record_output);
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bool_record_output",
      (void*)&__quantum__rt__bool_record_output);

  return runOrcJIT(argv[0]);
}
