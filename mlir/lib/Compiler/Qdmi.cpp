#include "mlir/Compiler/Qdmi.h"

#include "fomac/FoMaC.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

#include <memory>

void mlir::listAvailableQDMIDevices(fomac::Session& session,
                                    llvm::raw_ostream& os) {
  os << "Available QDMI devices:\n";
  for (auto dev : session.getDevices()) {
    os << '\t' << dev.getName() << '\n';
  }
}

std::shared_ptr<fomac::Session::Device>
mlir::getQDMIDevice(fomac::Session& session, StringRef name) {
  const auto devices = session.getDevices();

  auto it = devices.begin();
  for (; it != devices.end(); ++it) {
    if (it->getName() == name) {
      break;
    }
  }

  if (it == devices.end()) {
    return nullptr;
  }

  return std::make_shared<fomac::Session::Device>(*it);
}