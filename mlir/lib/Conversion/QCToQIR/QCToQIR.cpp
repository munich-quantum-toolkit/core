#include "mlir/Conversion/QCToQIR/QCToQIR.h"

#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"

#include <mlir/Pass/PassManager.h>

void mlir::populateQIRConversionPipeline(mlir::PassManager& pm,
                                         bool useAdaptive) {
  if (useAdaptive) {
    pm.addPass(createQCToQIRAdaptive());
  } else {
    pm.addPass(createQCToQIRBase());
  }
  pm.addPass(createAttachEntryPointAttributes(
      qir::AttachEntryPointAttributesOptions{useAdaptive}));
}