# Powered matrix numerical invariant

The probe ran on 2026-10-25 against Core base
`1d384a5d23aa1f02b26fd5684dab586ef26167fe` plus the uncommitted native
verification implementation **before** the resolved-matrix validation guard. It
used the existing release libraries; the exact compiler/linker argument arrays
and working directories are in `commands.json`. This is an implementation
finding, not a claim that the base revision had a public native verification
API.

`qco.unitary` accepts `diag(a,a)` for `a = 1.00000000004`: its entrywise
Gram-matrix deviation is approximately `8e-11`, within the existing `1e-10`
matrix-verifier tolerance. `qco.pow` raises the eigenvalues directly. Its check
of the eigenvectors' orthogonality does not constrain eigenvalue moduli or check
the finite/unitary invariant after exponentiation. A tolerated radial error can
therefore grow into a matrix that is not an isometry.

| Exponent | Expected physical invariant                | Observed before guard                                                                                                                            |
| -------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `1e10`   | Resolved matrix must be finite and unitary | Parsing succeeds; trace status `Complete`; radius `1.4918247470029882`; squared residual `0.24189158176455333`; old formula `0.4767218239523065` |
| `1e12`   | Resolved matrix must be finite and unitary | Parsing succeeds; trace status `Complete`; radius `2.3538604568457251e17`; squared residual `5.5406590503019657e34`; old formula `0`             |
| `1e14`   | Resolved matrix must be finite and unitary | Parsing succeeds; trace status `Complete`; radius and squared residual `NaN`; old formula `0`                                                    |

`before-guard.txt` is the raw output. The harness calls the actual parser and
execution resolver, then independently evaluates the implementation's old
`sqrt(max(0, r*(1-r/4)))` expression. It does **not** call `verify`; the
false-equivalence risk is inferred from the observed accepted trace and that
formula. In particular, the `1e10` case shows that merely rejecting `r > 2` does
not establish the isometry precondition for process distance.

The implementation now checks each resolved dense local matrix with the existing
entrywise Gram tolerance before any DD worker uses it, without allocating
another dense matrix. The loop polls cancellation/deadline between column pairs.
The process-distance calculation independently checks finite residuals and the
phase-aligned isometry range instead of clamping an arbitrary invalid radicand
to zero. Numerical uncertainty still includes the separate DD/floating-point
comparison guard; the existing matrix acceptance tolerance is not the user
synthesis-error budget.

Regression command, from the implementation checkout:

```sh
build/release/mlir/unittests/Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils --gtest_filter=DenseVerification.PowerCannotAmplifyAcceptedMatrixRoundingIntoNonunitarity
```

Expected after guard: all three programs still parse; `resolveExecution` returns
`Unsupported`; `verify` returns `Unsupported` for Exact and ProcessDistance with
Construction, Alternating and Auto. This covers both finite amplification and
overflow. Observed: the focused command passed on 2026-10-25; its complete
output is retained in `after-guard.txt`. The complete native-suite outcome is
recorded separately; this focused check does not stand in for that broader
validation.

Relevant source locations at the uncommitted implementation snapshot:

- `mlir/lib/Dialect/MQT/Utils/DenseUnitary.cpp:39`: original finite/Gram
  validator.
- `mlir/include/mqt/Dialect/MQT/Utils/DenseUnitary.h:27`: shared `1e-10`
  tolerance.
- `mlir/lib/Dialect/QCO/IR/Modifiers/PowOp.cpp:739`: existing eigendecomposition
  power implementation; no production edit in this investigation.
- `mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp:308`: resolved local matrix
  validation.
- `mlir/lib/Dialect/QCO/Utils/Verification.cpp:327`: finite residual and
  isometry-range guards.
- `mlir/unittests/Dialect/QCO/Utils/test_verification_dense.cpp:635`: native
  regression.

A separate change to the owning `PowOp` numerical invariant is an extraction
candidate: preventing radial-error amplification there would protect its other
consumers too. The verification implementation validates its own accepted
execution trace rather than broadening this task into a power-operation fix.

The exact standalone harness follows. To reproduce the original build command,
extract it to `/tmp/native-power-probe.cpp`; the include and link paths in
`commands.json` identify the precise original environment. There is no
additional production/test target for this exploratory harness; the permanent
regression above exercises the public verification API.

```cpp
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
int main() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::qco::QCODialect>();
  for (double exponent : {1e10, 1e12, 1e14}) {
    std::ostringstream source;
    source << std::scientific << std::setprecision(17)
      << "module { func.func @main() { %p = arith.constant " << exponent << R"( : f64
      %q = qco.static 0 : !qco.qubit
      %out = qco.pow(%p) (%a = %q) {
        %r = qco.unitary dense<[[(1.00000000004, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.00000000004, 0.0)]]> : tensor<2x2xcomplex<f64>> %a : !qco.qubit
        qco.yield %r : !qco.qubit
      } : {!qco.qubit} -> {!qco.qubit}
      qco.sink %out : !qco.qubit
      return
    } })";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source.str(), &context);
    std::cout << std::setprecision(17) << "exponent=" << exponent << " parsed=" << bool(module);
    if (!module) { std::cout << '\n'; continue; }
    const auto result = mlir::qco::resolveExecution(module->lookupSymbol<mlir::func::FuncOp>("main"));
    std::cout << " status=" << int(result.status) << " gates=" << result.gates.size();
    if (!result.gates.empty()) {
      auto radius = result.gates.front().matrix.front().real();
      auto residual = (radius - 1) * (radius - 1);
      auto oldDistance = std::sqrt(std::max(0., residual * (1 - residual / 4)));
      std::cout << " radius=" << radius << " squaredResidual=" << residual << " oldProcessDistance=" << oldDistance;
    }
    std::cout << " message=" << result.message << '\n';
  }
}
```
