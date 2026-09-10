# Reproduce the MLIR audit probes

Use a disposable checkout at `d994fe6833b6b7a9b1bccdeccc64e31c5c09ffd1` with
this evidence directory copied into it. Commands run from the repository root.
Build with LLVM/MLIR 23.1.0 and the normal Release preset as described in the
parent audit. Enable tests. Serialize builds and restore each experimental patch
before the next probe.

The patches are experimental evidence. They are not complete fixes: in
particular, the attribute patch leaves other comparator defects in place, and
the modifier patch changes the test call sites in place to measure ownership
before a maintainer decides how to organize the tests.

## Shared oracle and C++ ownership

```sh
git apply .agent/audits/mlir-tests-diagnostics/oracle-and-lifetime.patch
cmake --build --preset release --target mqt-core-mlir-unittests-compiler -j 8
./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
  --gtest_filter='Audit2254Oracle.*'
```

Baseline result: all seven input pairs parse and pass ordinary verification and
QCO linearity. All seven negative equivalence assertions fail. Exit status 1.

Run the ownership case separately because it terminates the process:

```sh
./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
  --gtest_filter='Audit2254Ownership.*'
```

Baseline result: SIGSEGV during independent-context move assignment, shell
status 139 on the tested machine. Test the lifetime-order explanation:

```sh
git apply .agent/audits/mlir-tests-diagnostics/lifetime-order.patch
cmake --build --preset release --target mqt-core-mlir-unittests-compiler -j 8
./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
  --gtest_filter='-Audit2254Oracle.*'
```

Result: exit 0, 205 tests pass. Restore both changes and rebuild:

```sh
git apply -R .agent/audits/mlir-tests-diagnostics/lifetime-order.patch
git apply -R .agent/audits/mlir-tests-diagnostics/oracle-and-lifetime.patch
cmake --build --preset release --target mqt-core-mlir-unittests-compiler -j 8
```

## Smaller, stricter attribute comparison

```sh
git apply .agent/audits/mlir-tests-diagnostics/attribute-comparison.patch
git apply .agent/audits/mlir-tests-diagnostics/oracle-and-lifetime.patch
cmake --build --preset release --target mqt-core-mlir-unittest-qco-ir \
  mqt-core-mlir-unittests-compiler -j 8
./build/release/mlir/unittests/Dialect/QCO/IR/mqt-core-mlir-unittest-qco-ir
./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
  --gtest_filter='Audit2254Oracle.DenseConstantValueMatters:Audit2254Oracle.FunctionArgumentTypeMatters'
```

Result: the 579 QCO tests and the two new negative cases pass. The helper is 76
lines shorter. The initial experiment changed only the helper and ran the full
CTest suite: only the grouped QCO entry failed, specifically
`QCOTest.IndexSwitchParser`. The saved patch also aligns that fixture's CBit
initialization with its reference. Both versions write all bits before reading
or returning them. No arbitrary attribute-ignore rule is retained for that
fixture difference. The retained floating tolerance is still unsound for general
classical comparisons; see the parent audit. This patch demonstrates a smaller,
stronger helper, not complete semantic equivalence.

```sh
git apply -R .agent/audits/mlir-tests-diagnostics/oracle-and-lifetime.patch
git apply -R .agent/audits/mlir-tests-diagnostics/attribute-comparison.patch
cmake --build --preset release -j 8
```

## Modifier verification ownership

```sh
git apply .agent/audits/mlir-tests-diagnostics/modifier-ownership.patch
cmake --build --preset release --target mqt-core-mlir-unittest-qc-to-qco -j 8
./build/release/mlir/unittests/Dialect/QC/IR/mqt-core-mlir-unittest-qc-ir
./build/release/mlir/unittests/Conversion/QCToQCO/mqt-core-mlir-unittest-qc-to-qco
```

Result: exit 0 from both binaries; 367 QC and 178 QC-to-QCO cases pass. The
malformed modifier matrices retain their diagnostic assertions but invoke the
owning verifier. The converter loses one whole-IR walk. The patch removes 26 net
lines, including 14 production lines.

```sh
git apply -R .agent/audits/mlir-tests-diagnostics/modifier-ownership.patch
cmake --build --preset release --target mqt-core-mlir-unittest-qc-to-qco -j 8
```

## Valid unsupported QCO input

```sh
git apply .agent/audits/mlir-tests-diagnostics/valid-conversion-input.patch
cmake --build --preset release --target mqt-core-mlir-unittest-qco-to-qc -j 8
./build/release/mlir/unittests/Conversion/QCOToQC/mqt-core-mlir-unittest-qco-to-qc
git apply -R .agent/audits/mlir-tests-diagnostics/valid-conversion-input.patch
cmake --build --preset release --target mqt-core-mlir-unittest-qco-to-qc -j 8
```

Result: exit 0, 153 tests pass. The corrected fixture passes ordinary
verification and linearity, then fails conversion for its missing trailing qubit
result, with the expected reason.

## Driver debugging

```sh
audit_cc=./build/release/mlir/tools/mqt-cc/mqt-cc
audit_data=.agent/audits/mlir-tests-diagnostics
mkdir -p build/audit-2254-output

"$audit_cc" "$audit_data/reduced.mlir" --emit=qco-optimized \
  --pass-pipeline='builtin.module(canonicalize,cse)'

"$audit_cc" "$audit_data/reduced.mlir" --emit=qco-optimized \
  --pass-pipeline='builtin.module(hadamard-lifting)' \
  --mlir-print-ir-before-all --mlir-disable-threading

"$audit_cc" "$audit_data/reduced.mlir" --emit=qco --mlir-print-op-generic

"$audit_cc" "$audit_data/unsupported.mlir" \
  --mlir-print-ir-before-all --mlir-print-ir-after-failure \
  --mlir-disable-threading \
  --mlir-pass-pipeline-crash-reproducer=build/audit-2254-output/failure.mlir

"$audit_cc" "$audit_data/unsupported.mlir" \
  --mlir-print-stacktrace-on-diagnostic --mlir-disable-threading

"$audit_cc" "$audit_data/reduced.mlir" --emit=mlir \
  --debug-only=dialect-conversion --mlir-disable-threading

"$audit_cc" "$audit_data/input.qasm" --emit=jeff \
  -o build/audit-2254-output/input.jeff
"$audit_cc" build/audit-2254-output/input.jeff --emit=qco \
  --mlir-print-ir-before-all --mlir-disable-threading

"$audit_cc" "$audit_data/failure.mlir" --emit=qc-import
"$audit_cc" "$audit_data/reduced.mlir" --emit=qco-optimized \
  --pass-pipeline='builtin.module(qc-to-qco)'
```

Observed results, in order:

1. Exit 1: `canonicalize` is not registered in the CLI.
2. Exit 0: HadamardLifting runs first; six cleanup passes follow it.
3. Exit 0: generic IR appears on stdout.
4. Exit 1: before/failure dumps appear and a reproducer file is written.
5. Exit 1: source location and primary error appear, but no stacktrace note.
6. Exit 0 with dialect-conversion tracing on the tested logging-enabled LLVM.
7. Export succeeds; reimport succeeds but requested initial-pass dump is absent.
8. Exit 0: the saved failing pipeline is not restored by ordinary file parsing.
9. Exit 1: `qc-to-qco` is not registered for textual replay.

Also compare `--passes=hadamard-lifting`, `--measurement-lifting`, and
`--hadamard-lifting` with the working textual form. The first two fail; the last
prints help and exits 0. Inspect output, not just exit status.

## Python consumed-state probes

Run each action in a fresh Python process using the rebuilt baseline package:

```python
import subprocess
import sys

setup = """from mqt.core.mlir import QCProgram, compile_program
p = QCProgram.from_qasm_str("OPENQASM 3.0; qubit q;")
p.to_qco()
assert not p.is_valid
"""
for action in ("p.copy()", "p.cleanup()", "compile_program(p)", "p.ir"):
    result = subprocess.run([sys.executable, "-c", setup + action], capture_output=True, check=False)
    print(action, result.returncode, result.stderr.decode())
```

The first three terminate with SIGABRT (`returncode == -6` on the tested
platform). The last raises `RuntimeError` and exits 1 normally.

## Python diagnostic-handler restoration

Run the following after installing the rebuilt baseline package. Capture stderr
and check the module diagnostic `failed to parse pass pipeline`:

```python
from mqt.core.mlir import (
    CompilerTarget,
    PayloadFormat,
    PayloadSpecification,
    QCProgram,
    TargetEnvironment,
)

program = QCProgram.from_qasm_str("""OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit c;
h q;
c = measure q;
if (c) { x q; }
""").to_qco()
valid = program.copy()
target = CompilerTarget(
    1,
    connectivity=CompilerTarget.Connectivity.all_to_all(),
    native_operations=CompilerTarget.NativeOperations.unrestricted(),
)
payload = PayloadSpecification(PayloadFormat("openqasm", "3.0"), [])
try:
    program.compile_for_target(TargetEnvironment(target, payload))
except RuntimeError as error:
    assert "qco.if" in str(error)
else:
    raise AssertionError("Expected target rejection")

try:
    valid.run_pass_pipeline("not-a-pass")
except RuntimeError:
    pass
else:
    raise AssertionError("Expected pipeline rejection")
```

Result: exit 0 and the expected module diagnostic on stderr. This checks actual
routing after the first handler is destroyed; the existing success-only action
does not.
