/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Dialect/QC/IR/QCDialect.h"

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "qir_programs.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

namespace mqt::test {

struct QIRConversionCase {
  const char* name;
  NamedMLIRBuilder<mlir::qc::QCProgramBuilder> programBuilder;
  NamedMLIRBuilder<mlir::qir::QIRProgramBuilder> referenceBuilder;
};

using QIRConversionTest = ::mqt::test::TestWithMLIRContext<
    QIRConversionCase, mlir::qc::QCDialect, mlir::LLVM::LLVMDialect,
    mlir::arith::ArithDialect, mlir::func::FuncDialect,
    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
    mlir::cf::ControlFlowDialect>;

constexpr QIRConversionCase qirCase(const char* name, auto program,
                                    auto reference) {
  return {name, {name, program}, {name, reference}};
}

inline constexpr auto COMMON_QIR_CASES = std::to_array<QIRConversionCase>({
    qirCase("Barrier", mlir::qc::barrier, mlir::qir::alloc1QubitRegister<true>),
    qirCase("BarrierTwoQubits", mlir::qc::barrierTwoQubits,
            mlir::qir::allocQubitRegister<true>),
    qirCase("BarrierMultipleQubits", mlir::qc::barrierMultipleQubits,
            mlir::qir::alloc3QubitRegister<true>),
    qirCase("SingleControlledBarrier", mlir::qc::singleControlledBarrier,
            mlir::qir::allocQubitRegister<true>),
    qirCase("DCX", mlir::qc::dcx, mlir::qir::dcx<true>),
    qirCase("SingleControlledDCX", mlir::qc::singleControlledDcx,
            mlir::qir::singleControlledDcx<true>),
    qirCase("MultipleControlledDCX", mlir::qc::multipleControlledDcx,
            mlir::qir::multipleControlledDcx<true>),
    qirCase("ECR", mlir::qc::ecr, mlir::qir::ecr<true>),
    qirCase("SingleControlledECR", mlir::qc::singleControlledEcr,
            mlir::qir::singleControlledEcr<true>),
    qirCase("MultipleControlledECR", mlir::qc::multipleControlledEcr,
            mlir::qir::multipleControlledEcr<true>),
    qirCase("GlobalPhase", mlir::qc::globalPhase, mlir::qir::globalPhase<true>),
    qirCase("H", mlir::qc::h, mlir::qir::h<true>),
    qirCase("SingleControlledH", mlir::qc::singleControlledH,
            mlir::qir::singleControlledH<true>),
    qirCase("MultipleControlledH", mlir::qc::multipleControlledH,
            mlir::qir::multipleControlledH<true>),
    qirCase("Identity", mlir::qc::identity, mlir::qir::identity<true>),
    qirCase("SingleControlledIdentity", mlir::qc::singleControlledIdentity,
            mlir::qir::twoQubitsOneIdentity<true>),
    qirCase("MultipleControlledIdentity", mlir::qc::multipleControlledIdentity,
            mlir::qir::threeQubitsOneIdentity<true>),
    qirCase("iSWAP", mlir::qc::iswap, mlir::qir::iswap<true>),
    qirCase("SingleControllediSWAP", mlir::qc::singleControlledIswap,
            mlir::qir::singleControlledIswap<true>),
    qirCase("MultipleControllediSWAP", mlir::qc::multipleControlledIswap,
            mlir::qir::multipleControlledIswap<true>),
    qirCase("P", mlir::qc::p, mlir::qir::p<true>),
    qirCase("SingleControlledP", mlir::qc::singleControlledP,
            mlir::qir::singleControlledP<true>),
    qirCase("MultipleControlledP", mlir::qc::multipleControlledP,
            mlir::qir::multipleControlledP<true>),
    qirCase("RCCX", mlir::qc::rccx, mlir::qir::rccx<true>),
    qirCase("SingleControlledRCCX", mlir::qc::singleControlledRccx,
            mlir::qir::singleControlledRccx<true>),
    qirCase("MultipleControlledRCCX", mlir::qc::multipleControlledRccx,
            mlir::qir::multipleControlledRccx<true>),
    qirCase("R", mlir::qc::r, mlir::qir::r<true>),
    qirCase("SingleControlledR", mlir::qc::singleControlledR,
            mlir::qir::singleControlledR<true>),
    qirCase("MultipleControlledR", mlir::qc::multipleControlledR,
            mlir::qir::multipleControlledR<true>),
    qirCase("RX", mlir::qc::rx, mlir::qir::rx<true>),
    qirCase("SingleControlledRX", mlir::qc::singleControlledRx,
            mlir::qir::singleControlledRx<true>),
    qirCase("MultipleControlledRX", mlir::qc::multipleControlledRx,
            mlir::qir::multipleControlledRx<true>),
    qirCase("RXX", mlir::qc::rxx, mlir::qir::rxx<true>),
    qirCase("SingleControlledRXX", mlir::qc::singleControlledRxx,
            mlir::qir::singleControlledRxx<true>),
    qirCase("MultipleControlledRXX", mlir::qc::multipleControlledRxx,
            mlir::qir::multipleControlledRxx<true>),
    qirCase("RY", mlir::qc::ry, mlir::qir::ry<true>),
    qirCase("SingleControlledRY", mlir::qc::singleControlledRy,
            mlir::qir::singleControlledRy<true>),
    qirCase("MultipleControlledRY", mlir::qc::multipleControlledRy,
            mlir::qir::multipleControlledRy<true>),
    qirCase("RYY", mlir::qc::ryy, mlir::qir::ryy<true>),
    qirCase("SingleControlledRYY", mlir::qc::singleControlledRyy,
            mlir::qir::singleControlledRyy<true>),
    qirCase("MultipleControlledRYY", mlir::qc::multipleControlledRyy,
            mlir::qir::multipleControlledRyy<true>),
    qirCase("RZ", mlir::qc::rz, mlir::qir::rz<true>),
    qirCase("SingleControlledRZ", mlir::qc::singleControlledRz,
            mlir::qir::singleControlledRz<true>),
    qirCase("MultipleControlledRZ", mlir::qc::multipleControlledRz,
            mlir::qir::multipleControlledRz<true>),
    qirCase("RZX", mlir::qc::rzx, mlir::qir::rzx<true>),
    qirCase("SingleControlledRZX", mlir::qc::singleControlledRzx,
            mlir::qir::singleControlledRzx<true>),
    qirCase("MultipleControlledRZX", mlir::qc::multipleControlledRzx,
            mlir::qir::multipleControlledRzx<true>),
    qirCase("RZZ", mlir::qc::rzz, mlir::qir::rzz<true>),
    qirCase("SingleControlledRZZ", mlir::qc::singleControlledRzz,
            mlir::qir::singleControlledRzz<true>),
    qirCase("MultipleControlledRZZ", mlir::qc::multipleControlledRzz,
            mlir::qir::multipleControlledRzz<true>),
    qirCase("S", mlir::qc::s, mlir::qir::s<true>),
    qirCase("SingleControlledS", mlir::qc::singleControlledS,
            mlir::qir::singleControlledS<true>),
    qirCase("MultipleControlledS", mlir::qc::multipleControlledS,
            mlir::qir::multipleControlledS<true>),
    qirCase("Sdg", mlir::qc::sdg, mlir::qir::sdg<true>),
    qirCase("SingleControlledSdg", mlir::qc::singleControlledSdg,
            mlir::qir::singleControlledSdg<true>),
    qirCase("MultipleControlledSdg", mlir::qc::multipleControlledSdg,
            mlir::qir::multipleControlledSdg<true>),
    qirCase("SWAP", mlir::qc::swap, mlir::qir::swap<true>),
    qirCase("SingleControlledSWAP", mlir::qc::singleControlledSwap,
            mlir::qir::singleControlledSwap<true>),
    qirCase("MultipleControlledSWAP", mlir::qc::multipleControlledSwap,
            mlir::qir::multipleControlledSwap<true>),
    qirCase("SX", mlir::qc::sx, mlir::qir::sx<true>),
    qirCase("SingleControlledSX", mlir::qc::singleControlledSx,
            mlir::qir::singleControlledSx<true>),
    qirCase("MultipleControlledSX", mlir::qc::multipleControlledSx,
            mlir::qir::multipleControlledSx<true>),
    qirCase("SXdg", mlir::qc::sxdg, mlir::qir::sxdg<true>),
    qirCase("SingleControlledSXdg", mlir::qc::singleControlledSxdg,
            mlir::qir::singleControlledSxdg<true>),
    qirCase("MultipleControlledSXdg", mlir::qc::multipleControlledSxdg,
            mlir::qir::multipleControlledSxdg<true>),
    qirCase("T", mlir::qc::t_, mlir::qir::t_<true>),
    qirCase("SingleControlledT", mlir::qc::singleControlledT,
            mlir::qir::singleControlledT<true>),
    qirCase("MultipleControlledT", mlir::qc::multipleControlledT,
            mlir::qir::multipleControlledT<true>),
    qirCase("Tdg", mlir::qc::tdg, mlir::qir::tdg<true>),
    qirCase("SingleControlledTdg", mlir::qc::singleControlledTdg,
            mlir::qir::singleControlledTdg<true>),
    qirCase("MultipleControlledTdg", mlir::qc::multipleControlledTdg,
            mlir::qir::multipleControlledTdg<true>),
    qirCase("U2", mlir::qc::u2, mlir::qir::u2<true>),
    qirCase("SingleControlledU2", mlir::qc::singleControlledU2,
            mlir::qir::singleControlledU2<true>),
    qirCase("MultipleControlledU2", mlir::qc::multipleControlledU2,
            mlir::qir::multipleControlledU2<true>),
    qirCase("U", mlir::qc::u, mlir::qir::u<true>),
    qirCase("SingleControlledU", mlir::qc::singleControlledU,
            mlir::qir::singleControlledU<true>),
    qirCase("MultipleControlledU", mlir::qc::multipleControlledU,
            mlir::qir::multipleControlledU<true>),
    qirCase("X", mlir::qc::x, mlir::qir::x<true>),
    qirCase("SingleControlledX", mlir::qc::singleControlledX,
            mlir::qir::singleControlledX<true>),
    qirCase("MultipleControlledX", mlir::qc::multipleControlledX,
            mlir::qir::multipleControlledX<true>),
    qirCase("XXMinusYY", mlir::qc::xxMinusYY, mlir::qir::xxMinusYY<true>),
    qirCase("SingleControlledXXMinusYY", mlir::qc::singleControlledXxMinusYY,
            mlir::qir::singleControlledXxMinusYY<true>),
    qirCase("MultipleControlledXXMinusYY",
            mlir::qc::multipleControlledXxMinusYY,
            mlir::qir::multipleControlledXxMinusYY<true>),
    qirCase("XXPlusYY", mlir::qc::xxPlusYY, mlir::qir::xxPlusYY<true>),
    qirCase("SingleControlledXXPlusYY", mlir::qc::singleControlledXxPlusYY,
            mlir::qir::singleControlledXxPlusYY<true>),
    qirCase("MultipleControlledXXPlusYY", mlir::qc::multipleControlledXxPlusYY,
            mlir::qir::multipleControlledXxPlusYY<true>),
    qirCase("Y", mlir::qc::y, mlir::qir::y<true>),
    qirCase("SingleControlledY", mlir::qc::singleControlledY,
            mlir::qir::singleControlledY<true>),
    qirCase("MultipleControlledY", mlir::qc::multipleControlledY,
            mlir::qir::multipleControlledY<true>),
    qirCase("Z", mlir::qc::z, mlir::qir::z<true>),
    qirCase("SingleControlledZ", mlir::qc::singleControlledZ,
            mlir::qir::singleControlledZ<true>),
    qirCase("MultipleControlledZ", mlir::qc::multipleControlledZ,
            mlir::qir::multipleControlledZ<true>),
    qirCase("SingleMeasurementToSingleBit",
            mlir::qc::singleMeasurementToSingleBit,
            mlir::qir::singleMeasurementToSingleBit),
    qirCase("MultipleClassicalRegistersAndMeasurements",
            mlir::qc::multipleClassicalRegistersAndMeasurements,
            mlir::qir::multipleClassicalRegistersAndMeasurements),
    qirCase("AllocQubit", mlir::qc::allocQubit, mlir::qir::allocQubit<true>),
    qirCase("AllocQubitRegister", mlir::qc::allocQubitRegister,
            mlir::qir::allocQubitRegister<true>),
    qirCase("AllocMultipleQubitRegisters",
            mlir::qc::allocMultipleQubitRegisters,
            mlir::qir::allocMultipleQubitRegisters<true>),
    qirCase("AllocMultipleQubitRegistersWithOps",
            mlir::qc::allocMultipleQubitRegistersWithOps,
            mlir::qir::allocMultipleQubitRegistersWithOps<true>),
    qirCase("AllocLargeRegister", mlir::qc::allocLargeRegister,
            mlir::qir::allocQubitRegister<true>),
    qirCase("StaticQubits", mlir::qc::staticQubits, mlir::qir::staticQubits),
    qirCase("StaticQubitsWithOps", mlir::qc::staticQubitsWithOps,
            mlir::qir::staticQubitsWithOps),
    qirCase("StaticQubitsWithParametricOps",
            mlir::qc::staticQubitsWithParametricOps,
            mlir::qir::staticQubitsWithParametricOps),
    qirCase("StaticQubitsWithTwoTargetOps",
            mlir::qc::staticQubitsWithTwoTargetOps,
            mlir::qir::staticQubitsWithTwoTargetOps),
    qirCase("StaticQubitsWithCtrl", mlir::qc::staticQubitsWithCtrl,
            mlir::qir::staticQubitsWithCtrl),
    qirCase("StaticQubitsWithInv", mlir::qc::staticQubitsWithInv,
            mlir::qir::staticQubitsWithInv),
    qirCase("AllocDeallocPair", mlir::qc::allocDeallocPair,
            mlir::qir::emptyQIR<true>),
    qirCase("CtrlTwo", mlir::qc::ctrlTwo, mlir::qir::ctrlTwo<true>),
});

inline bool verifyQIRTestStage(mlir::ModuleOp moduleOp, llvm::StringRef stage) {
  const bool valid = mlir::succeeded(mlir::verify(moduleOp));
  if (!valid || irPrintingForced()) {
    llvm::errs() << stage << ":\n";
    moduleOp.print(llvm::errs());
    llvm::errs() << '\n';
  }
  return valid;
}

inline void expectEquivalentQIR(mlir::ModuleOp actual,
                                mlir::ModuleOp expected) {
  if (areModulesEquivalentWithPermutations(actual, expected)) {
    return;
  }
  ADD_FAILURE() << "Converted QIR differs from the reference";
  if (!irPrintingForced()) {
    llvm::errs() << "Converted QIR:\n";
    actual.print(llvm::errs());
    llvm::errs() << "\nReference QIR:\n";
    expected.print(llvm::errs());
    llvm::errs() << '\n';
  }
}

} // namespace mqt::test
