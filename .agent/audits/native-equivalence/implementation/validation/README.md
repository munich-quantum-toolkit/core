# DD, borrowed-program and Python validation

Executed 2026-10-25 in the native-equivalence implementation checkout based on
Core `1d384a5d23aa1f02b26fd5684dab586ef26167fe`. Implementation changes are
uncommitted. This directory records local checks, not hosted CI.

| Check                                     | Command                                                                                                        | Expected                                                                                                             | Observed                                                                                                             |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| DD suite including cooperative resources  | `cmake --build /tmp/native-equivalence-dd-build -j4 && /tmp/native-equivalence-dd-build/dd-all-test`           | Existing arithmetic unchanged; cancellation and memory failures unwind and release counted storage                   | 220/220 passed, 283 ms; `dd.log`                                                                                     |
| Generated Python stubs and rebuilt wheel  | `CMAKE_BUILD_PARALLEL_LEVEL=4 uvx nox -s stubs`                                                                | Build/import/stub generation succeeds                                                                                | Passed; only `python/mqt/core/mlir.pyi` differs; `stubs.log`                                                         |
| Verification and existing DD Python tests | `.nox/stubs/bin/python -m pytest -o addopts='' test/python/test_verification.py test/python/test_qco_dd.py`    | Bound SCF, phase, lifetime, typed errors, simulation/resource results, concurrent calls; existing DD tests unchanged | 34/34 passed; `python-verification-dd.log`                                                                           |
| Borrowed C++ program tests                | `build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter=CompilerVerification.*` | Borrow inputs, reject invalid binding/context/moved owners, require marked entry                                     | 3/3 passed in the runtime agent's broader native run, `/tmp/native-equivalence-final-physical-fixtures-compiler.log` |
| Pinned QCEC differential                  | Commands in `../qcec-differential/README.md`                                                                   | Eleven declared equivalences/differences, including expected phase and numerical-guard contract differences          | All expected; final native executable hashes and embedded exact inputs retained there                                |

The standalone DD configuration is preserved in `dd-CMakeLists.txt`. It uses the
current source files, existing generated version headers and downloaded
GoogleTest/JSON dependencies from the normal release configuration. Name the
preserved file `CMakeLists.txt` in that source directory and configure with:

```sh
cmake -S /tmp/native-equivalence-dd-src \
  -B /tmp/native-equivalence-dd-build -G Ninja
```

The isolated target does not link MLIR or the new verifier. The resource tests
cover preallocation cancellation/deadline/one-byte rejection, lazy cache
allocation, mid-recursion cancellation, cached and terminal arithmetic, shared
aggregate allocation budgets, and independent cancellation controls.

Before the focused Python tests, the nox-managed environment was supplemented
with `uv pip install --python .nox/stubs/bin/python pytest numpy qiskit==2.5.2`.
The QCEC dependency environment was neither replaced nor upgraded.

The DD byte limit counts allocator requests through the supplied resource,
including simultaneously live replacement allocations. It is not an RSS cap.
Stack storage, allocator overhead, and containers not using the resource are
excluded. Deadlines are cooperative; garbage-collection mark/sweep remains an
indivisible operation. The verifier must destroy a package after interrupted
arithmetic. External concurrent mutation of global DD tolerance remains outside
the existing DD contract; verification does not mutate that tolerance.

An initial native wrapper test exposed a test-only diagnostic-handler lifetime
error after moving its program owner. Retaining the context until the handler is
destroyed fixes that hang without weakening the moved-owner rejection.

The [powered-matrix regression](pow-amplification/README.md) retains the
independent amplification probe and before/after results for the resolved-matrix
unitarity guard.

## Local C++ lint environment

The final gate passed with **zero diagnostics**. It checked 29 files: 28 source
translation units and `mlir/unittests/VerificationTest.h`. The exact list is in
`cpp-lint-files.txt`; `cpp-lint.log` contains every command and all result
counts. `cpp-lint-source-manifest.json` records SHA-256 values for all 42
changed C++ paths, including the deleted helper, and confirms that both the path
set and contents were unchanged throughout the final run. The final run includes
the powered-matrix guard, numerical residual-bound fix, dense-reference tests,
and final fixed-physical compiler fixture corrections.

The repository gate is `uvx nox -s cpp-lint`. It uses Clang 23.1.4 and
`cpp-linter==1.15.0`, compares against `origin/main`, and checks complete
changed translation units (`--lines-changed-only=false`). Its existing ignore
rules exclude public include directories as standalone files; those headers are
parsed through their users. The generated-header CMake target runs first.

All new task C++ files were marked with `git add -N`. Because cpp-linter's local
diff discovery uses staged changes and does not discover intent-to-add content,
a separate index was populated with the 42 changed C++ paths:

```python
import os, subprocess

paths = subprocess.check_output(
    ["git", "diff", "--name-only", "origin/main", "--", "*.cpp", "*.h", "*.hpp"],
    text=True,
).splitlines()
env = dict(os.environ, GIT_INDEX_FILE="/tmp/native-equivalence-lint.index")
subprocess.run(["git", "read-tree", "HEAD"], env=env, check=True)
subprocess.run(["git", "add", "--", *paths], env=env, check=True)
```

The actual gate invocation is:

```sh
GIT_INDEX_FILE=/tmp/native-equivalence-lint.index CMAKE_BUILD_PARALLEL_LEVEL=4 uvx nox -s cpp-lint
```

This leaves the real index unstaged apart from the intent-to-add markers. No
commit was created. The selected paths include the new dense-reference test.

There is a separate local infrastructure issue: cpp-linter's local feedback
client logs its result counts but does not create the `GITHUB_OUTPUT` file that
the nox session reads, causing `FileNotFoundError` after lint completes. The
only workaround is inside the ignored `.nox/cpp-lint` environment: a wrapper
around `LinterClient.set_exit_code` calls the original function, then writes
those same three actual counts to `GITHUB_OUTPUT` when `client_kind == "local"`.
It changes neither checks nor counts nor exit status. No repository workflow,
nox session, checker configuration or source diagnostic suppression was added.
The shim's behavior was confirmed with the nonzero ten-finding pass: nox
reported exactly ten failures instead of the infrastructure exception.
